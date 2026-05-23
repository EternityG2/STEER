#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run stage-3 LLM-as-a-judge labeling for process supervision."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import ray
except ImportError as e:
    raise SystemExit("ray is not installed. Install it first.") from e

from openai import OpenAI
from tqdm import tqdm


SECTION_ORDER = [
    "Bug Analysis",
    "Test Design",
    "Output Prediction",
    "Input",
    "Expected Output",
]

JUDGE_SECTION_ORDER = [
    "Bug Analysis",
    "Test Design",
    "Output Prediction",
]

DEFAULT_STAGE3_USER_TEMPLATE = """I will provide a programming task description, a buggy code solution, and a candidate answer split into three steps.

Judge these three steps:
1. Bug Analysis
2. Test Design
3. Output Prediction

Rules:
- Judge correctness with reasonable tolerance, not perfection.
- Do not assume there is only one valid bug explanation.
- Judge each step independently, but also check whether Steps 2 and 3 are consistent with earlier steps.
- Only judge the three steps below. Do not judge the final Input or Expected Output sections.

How to judge each step:

Step 1: Bug Analysis
- Mark 1 if it identifies a real bug or flawed assumption in the code.
- It can be incomplete, but it must be supported by the code and useful for explaining wrong behavior.
- Mark 0 if the claimed bug is false, unsupported, or too generic to be useful.

Step 2: Test Design
- Mark 1 if the test is based on Step 1 and can expose the claimed bug.
- The test does not need to be fully finalized, but the trigger should be clear.
- Prefer valid task inputs; if it clearly violates important task constraints, usually mark 0.
- Mark 0 if the test would not actually trigger the claimed bug.

Step 3: Output Prediction
- Mark 1 if it matches Steps 1 and 2 and correctly states the expected correct output or result for the test.
- A concrete value is not required, but the expected correct outcome must be clear.
- Mark 0 if it only restates the task, is inconsistent with earlier steps, or does not describe the correct outcome for the test.

Output exactly in this format:
<analysis_1>...</analysis_1>
<label_1>0 or 1</label_1>
<analysis_2>...</analysis_2>
<label_2>0 or 1</label_2>
<analysis_3>...</analysis_3>
<label_3>0 or 1</label_3>
<first_incorrect_step>-1 or 1 or 2 or 3</first_incorrect_step>
<conclusion>Correct or Incorrect</conclusion>

[Task Description]
<task_description>
{description}
</task_description>

[Buggy Code]
<buggy_code>
{buggy_code}
</buggy_code>

[Candidate Answer]
<paragraph_1>
{bug_analysis}
</paragraph_1>
<paragraph_2>
{test_design}
</paragraph_2>
<paragraph_3>
{output_prediction}
</paragraph_3>

Now provide your review directly.
"""

@dataclass
class Stage3Config:
    input_jsonl: str
    output_dir: str

    api_base_url: str = os.getenv("JUDGE_API_BASE_URL", "")
    api_key: str = os.getenv("JUDGE_API_KEY", "")
    model_name: str = os.getenv("JUDGE_MODEL_NAME", "")
    temperature: float = 0.7
    max_tokens: int = 2048
    max_retries: int = 3

    ray_num_actors: int = 8
    ray_address: Optional[str] = None

    max_samples_per_task: int = 16
    max_judge_steps: int = 3

    flush_every: int = 1
    seed: int = 42
    save_raw: bool = True


class SectionHelper:
    @staticmethod
    def extract_all_sections(text: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
        if not text:
            return None

        result = {
            "Bug Analysis": None,
            "Test Design": None,
            "Output Prediction": None,
            "Input": None,
            "Expected Output": None,
        }

        def extract_block(start_header: str, end_header: Optional[str], content: str) -> Optional[str]:
            escaped_start = re.escape(start_header)
            if end_header:
                escaped_end = re.escape(end_header)
                pattern = (
                    rf"(?:###|##|\*\*|{escaped_start}:?)\s*{escaped_start}[:\s]*\n+(.*?)"
                    rf"(?=\n\s*(?:###|##|\*\*|{escaped_end}))"
                )
            else:
                pattern = rf"(?:###|##|\*\*|{escaped_start}:?)\s*{escaped_start}[:\s]*\n+(.*)"

            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return (
                    match.group(1)
                    .replace("```", "")
                    .replace("<code>", "")
                    .replace("</code>", "")
                    .strip()
                )
            return None

        result["Bug Analysis"] = extract_block("Bug Analysis", "Test Design", text)
        result["Test Design"] = extract_block("Test Design", "Output Prediction", text)
        result["Output Prediction"] = extract_block("Output Prediction", "Input", text)
        result["Input"] = extract_block("Input", "Expected Output", text)
        result["Expected Output"] = extract_block("Expected Output", None, text)
        return result


def sanitize_sections(sample: Dict[str, Any]) -> Optional[Dict[str, str]]:
    sections = sample.get("sections")
    if not sections:
        raw = sample.get("raw_response") or ""
        sections = SectionHelper.extract_all_sections(raw)
    if not sections:
        return None

    fixed: Dict[str, str] = {}
    for key in SECTION_ORDER:
        value = sections.get(key) if isinstance(sections, dict) else None
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        fixed[key] = value
    return fixed


def build_judge_user_prompt(task_row: Dict[str, Any], sections: Dict[str, str]) -> str:
    return DEFAULT_STAGE3_USER_TEMPLATE.format(
        description=task_row.get("description", ""),
        buggy_code=task_row.get("buggy_code", ""),
        bug_analysis=sections["Bug Analysis"],
        test_design=sections["Test Design"],
        output_prediction=sections["Output Prediction"],
    )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error line={line_no}: {e}") from e
    return rows


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def get_existing_task_ids(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tid = obj.get("task_id")
                if tid is not None:
                    seen.add(tid)
            except Exception:
                continue
    return seen


class JudgeClient:
    def __init__(self, api_base_url: str, api_key: str, model_name: str, temperature: float, max_tokens: int, max_retries: int):
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def judge_once(self, user_prompt: str) -> str:
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content or ""
                if content.strip():
                    return content
                last_err = RuntimeError("empty_response")
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"judge request failed: {last_err}")


def parse_judge_output(text: str) -> Tuple[bool, Dict[str, Any]]:
    def get_tag(tag: str) -> Optional[str]:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    analyses = []
    labels = []
    for i in range(1, 4):
        analyses.append(get_tag(f"analysis_{i}"))
        raw_label = get_tag(f"label_{i}")
        if raw_label is None:
            labels.append(None)
        else:
            raw_label = raw_label.strip()
            if raw_label in {"0", "1"}:
                labels.append(int(raw_label))
            else:
                labels.append(None)

    raw_first_err = get_tag("first_incorrect_step")
    conclusion = get_tag("conclusion")

    parsed = {
        "analyses": analyses,
        "labels": labels,
        "first_incorrect_step": None,
        "conclusion": conclusion,
    }

    if raw_first_err is not None:
        raw_first_err = raw_first_err.strip()
        if raw_first_err in {"-1", "1", "2", "3"}:
            parsed["first_incorrect_step"] = int(raw_first_err)

    parse_ok = all(a is not None for a in analyses) and all(l is not None for l in labels) and parsed["first_incorrect_step"] is not None and conclusion is not None
    if not parse_ok:
        return False, parsed

    labels_int = [int(x) for x in labels]
    first_err = parsed["first_incorrect_step"]

    if first_err == -1:
        norm_labels = [1, 1, 1]
        norm_conclusion = "Correct"
    else:
        norm_labels = [1 if i < first_err else 0 for i in range(1, 4)]
        norm_conclusion = "Incorrect"

    repaired = labels_int != norm_labels or (conclusion.strip().lower() not in {norm_conclusion.lower()})

    parsed["labels"] = norm_labels
    parsed["conclusion"] = norm_conclusion
    parsed["repaired"] = repaired
    return True, parsed


@ray.remote(num_cpus=1)
class Stage3Actor:
    def __init__(self, cfg_dict: Dict[str, Any]):
        self.cfg = Stage3Config(**cfg_dict)
        self.client = JudgeClient(
            api_base_url=self.cfg.api_base_url,
            api_key=self.cfg.api_key,
            model_name=self.cfg.model_name,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            max_retries=self.cfg.max_retries,
        )
        self.rng = random.Random(self.cfg.seed)

    def process_task(self, task_row: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task_row.get("task_id", "unknown_task")
        samples = task_row.get("samples", [])[: self.cfg.max_samples_per_task]

        raw_rows: List[Dict[str, Any]] = []
        sample_rows: List[Dict[str, Any]] = []

        num_valid_samples = 0
        num_judged_samples = 0
        num_parse_fail = 0
        num_repaired = 0

        for sample_idx, sample in enumerate(samples):
            sample_record: Dict[str, Any] = {
                "sample_idx": sample_idx,
                "orig_label": sample.get("label"),
                "orig_reward": sample.get("reward"),
                "response_md5": sample.get("response_md5"),
                "raw_response": sample.get("raw_response"),
                "judge_steps": [],
                "judge_first_error_step": None,
                "judge_all_correct": None,
                "judge_parse_ok": False,
                "judge_repaired": False,
                "skipped": False,
                "skip_reason": None,
            }

            if sample.get("skipped") is True:
                sample_record["skipped"] = True
                sample_record["skip_reason"] = f"upstream_{sample.get('skip_reason', 'skipped')}"
                sample_rows.append(sample_record)
                continue

            sections = sanitize_sections(sample)
            if not sections:
                sample_record["skipped"] = True
                sample_record["skip_reason"] = "sections_invalid_or_incomplete"
                sample_rows.append(sample_record)
                continue

            num_valid_samples += 1
            sample_record["sections"] = sections
            user_prompt = build_judge_user_prompt(task_row, sections)

            try:
                raw_text = self.client.judge_once(user_prompt)
            except Exception as e:
                sample_record["skipped"] = True
                sample_record["skip_reason"] = f"judge_request_failed: {e}"
                sample_rows.append(sample_record)
                continue

            parse_ok, parsed = parse_judge_output(raw_text)
            if not parse_ok:
                num_parse_fail += 1
                sample_record["skipped"] = True
                sample_record["skip_reason"] = "judge_output_parse_failed"
                sample_record["judge_raw_response"] = raw_text
                sample_record["judge_parsed_partial"] = parsed
                sample_rows.append(sample_record)
                if self.cfg.save_raw:
                    raw_rows.append({
                        "task_id": task_id,
                        "sample_idx": sample_idx,
                        "judge_raw_response": raw_text,
                        "judge_parsed_partial": parsed,
                        "parse_ok": False,
                    })
                continue

            num_judged_samples += 1
            if parsed.get("repaired"):
                num_repaired += 1

            labels = parsed["labels"]
            first_err = parsed["first_incorrect_step"]
            sample_record["judge_parse_ok"] = True
            sample_record["judge_repaired"] = bool(parsed.get("repaired", False))
            sample_record["judge_first_error_step"] = first_err
            sample_record["judge_all_correct"] = (first_err == -1)
            sample_record["judge_conclusion"] = parsed.get("conclusion")
            sample_record["judge_raw_response"] = raw_text

            for step_idx, step_name in enumerate(JUDGE_SECTION_ORDER[: self.cfg.max_judge_steps]):
                sample_record["judge_steps"].append({
                    "step_idx": step_idx,
                    "step_name": step_name,
                    "judge_label": int(labels[step_idx]),
                    "analysis": parsed["analyses"][step_idx],
                })

            sample_rows.append(sample_record)

            if self.cfg.save_raw:
                raw_rows.append({
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "judge_raw_response": raw_text,
                    "judge_first_error_step": first_err,
                    "judge_labels": labels,
                    "judge_conclusion": parsed.get("conclusion"),
                    "judge_repaired": bool(parsed.get("repaired", False)),
                    "parse_ok": True,
                })

        task_result = {
            "task_id": task_id,
            "description": task_row.get("description"),
            "buggy_code": task_row.get("buggy_code"),
            "buggy_lang": task_row.get("buggy_lang"),
            "oracle_code": task_row.get("oracle_code"),
            "oracle_lang": task_row.get("oracle_lang"),
            "base_score": task_row.get("base_score"),
            "processed_samples": len(samples),
            "valid_samples": num_valid_samples,
            "judged_samples": num_judged_samples,
            "judge_parse_fail_samples": num_parse_fail,
            "judge_repaired_samples": num_repaired,
            "judge_steps_computed": min(self.cfg.max_judge_steps, len(JUDGE_SECTION_ORDER)),
            "samples": sample_rows,
        }

        return {
            "task_result": task_result,
            "raw_rows": raw_rows if self.cfg.save_raw else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 LLM-as-a-judge labeling with Ray.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Usually stage2_mc_tasks.jsonl.")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--api_base_url", type=str, default=os.getenv("JUDGE_API_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--model_name", type=str, default=os.getenv("JUDGE_MODEL_NAME", ""))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_retries", type=int, default=3)

    parser.add_argument("--ray_num_actors", type=int, default=8)
    parser.add_argument("--ray_address", type=str, default=None)

    parser.add_argument("--max_samples_per_task", type=int, default=16)
    parser.add_argument("--max_judge_steps", type=int, default=3)
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_raw", action="store_true", default=True)
    parser.add_argument("--no_save_raw", dest="save_raw", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Stage3Config(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        ray_num_actors=args.ray_num_actors,
        ray_address=args.ray_address,
        max_samples_per_task=args.max_samples_per_task,
        max_judge_steps=args.max_judge_steps,
        flush_every=max(1, args.flush_every),
        seed=args.seed,
        save_raw=args.save_raw,
    )

    missing_api = [name for name in ["api_base_url", "api_key", "model_name"] if not getattr(cfg, name)]
    if missing_api:
        raise ValueError(f"Missing required judge API settings: {', '.join(missing_api)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_out_path = output_dir / "judge_tasks.jsonl"
    raw_out_path = output_dir / "judge_raw.jsonl"
    stats_path = output_dir / "stats.json"

    input_rows = load_jsonl(cfg.input_jsonl)
    if not input_rows:
        raise ValueError("input file is empty")

    existing_task_ids = get_existing_task_ids(tasks_out_path)
    tasks = [row for row in input_rows if row.get("task_id") not in existing_task_ids]

    if not tasks:
        print("All tasks are already processed.")
        return

    ray.init(address=cfg.ray_address, ignore_reinit_error=True)
    actors = [Stage3Actor.remote(asdict(cfg)) for _ in range(cfg.ray_num_actors)]

    in_flight: Dict[Any, Any] = {}
    task_iter = iter(tasks)
    for actor in actors:
        try:
            row = next(task_iter)
        except StopIteration:
            break
        ref = actor.process_task.remote(row)
        in_flight[ref] = actor

    processed_tasks = 0
    total_processed_samples = 0
    total_valid_samples = 0
    total_judged_samples = 0
    total_parse_fail_samples = 0
    total_repaired_samples = 0
    total_raw_rows = 0

    buffer_task_rows: List[Dict[str, Any]] = []
    buffer_raw_rows: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(tasks), desc="Stage-3 Judge")

    def flush_buffers(force: bool = False) -> None:
        nonlocal buffer_task_rows, buffer_raw_rows
        if not force and len(buffer_task_rows) < cfg.flush_every:
            return
        append_jsonl(tasks_out_path, buffer_task_rows)
        buffer_task_rows = []
        if cfg.save_raw:
            append_jsonl(raw_out_path, buffer_raw_rows)
            buffer_raw_rows = []

        stats = {
            "input_jsonl": cfg.input_jsonl,
            "remaining_run_tasks": len(tasks),
            "already_existing_tasks": len(existing_task_ids),
            "processed_tasks_in_this_run": processed_tasks,
            "processed_tasks_total_output_estimate": len(existing_task_ids) + processed_tasks,
            "total_processed_samples": total_processed_samples,
            "total_valid_samples": total_valid_samples,
            "total_judged_samples": total_judged_samples,
            "total_parse_fail_samples": total_parse_fail_samples,
            "total_repaired_samples": total_repaired_samples,
            "total_raw_rows": total_raw_rows if cfg.save_raw else None,
            "avg_judged_samples_per_task": (total_judged_samples / processed_tasks) if processed_tasks else 0.0,
            "config": asdict(cfg),
            "output_files": {
                "tasks": str(tasks_out_path),
                "raw": str(raw_out_path) if cfg.save_raw else None,
            },
        }
        write_json(stats_path, stats)

    while in_flight:
        done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        done_ref = done_refs[0]
        actor = in_flight.pop(done_ref)
        result = ray.get(done_ref)

        task_result = result["task_result"]
        raw_rows = result.get("raw_rows") or []

        processed_tasks += 1
        total_processed_samples += int(task_result.get("processed_samples", 0))
        total_valid_samples += int(task_result.get("valid_samples", 0))
        total_judged_samples += int(task_result.get("judged_samples", 0))
        total_parse_fail_samples += int(task_result.get("judge_parse_fail_samples", 0))
        total_repaired_samples += int(task_result.get("judge_repaired_samples", 0))
        total_raw_rows += len(raw_rows)

        buffer_task_rows.append(task_result)
        if cfg.save_raw and raw_rows:
            buffer_raw_rows.extend(raw_rows)

        flush_buffers(force=False)
        pbar.update(1)

        try:
            next_row = next(task_iter)
            new_ref = actor.process_task.remote(next_row)
            in_flight[new_ref] = actor
        except StopIteration:
            pass

    pbar.close()
    flush_buffers(force=True)
    print(f"[Done] stage3 judge results written to: {tasks_out_path}")
    if cfg.save_raw:
        print(f"[Done] stage3 raw judge replies written to: {raw_out_path}")
    print(f"[Done] stats written to: {stats_path}")


if __name__ == "__main__":
    main()
