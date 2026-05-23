#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute stage-2 step-level Monte Carlo scores."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import ray
except ImportError as e:
    raise SystemExit("ray is not installed. Install it first.") from e

from openai import OpenAI
from tqdm import tqdm


DEFAULT_PROMPT_TEMPLATE = (
    "You are given a programming task description and an **incorrect**\n"
    "code solution.\n"
    "Your goal is to identify the bug in the code and construct a **Hack Test Case** that triggers the failure of the solution.\n"
    "\n"
    "## Task Description\n"
    "{description}\n"
    "\n"
    "## Incorrect Code Solution\n"
    "```\n"
    "{code}\n"
    "```\n"
    "\n"
    "## Important Notes\n"
    "**DO NOT** copy or reuse the example test cases provided in the Task Description.\n"
    "## Response Format\n"
    "Your response MUST exactly follow the format below:\n"
    "\n"
    "### Bug Analysis\n"
    "Briefly describe the logic or implementation error in the code.\n"
    "\n"
    "### Test Design\n"
    "Based on the bug analysis, describe the input values chosen to trigger the bug.\n"
    "\n"
    "### Output Prediction\n"
    "Concisely derive the correct output based on the Task Description logic.\n"
    "### Input\n"
    "Provide the raw standard input that follows the exact input specification. Do NOT include ANY reasoning, explanations, or markdown formatting.\n"
    "\n"
    "### Expected Output\n"
    "Provide the correct standard output according to the task description. Do NOT include ANY reasoning, explanations, or markdown formatting.\n"
)

try:
    from prompts import Prompt as PROJECT_PROMPT_TEMPLATE  # type: ignore
except Exception:
    PROJECT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE


@dataclass
class Stage2Config:
    input_jsonl: str
    output_dir: str

    api_base_url: str = os.getenv("GENERATION_API_BASE_URL", "")
    api_key: str = os.getenv("GENERATION_API_KEY", "")
    model_name: str = os.getenv("GENERATION_MODEL_NAME", "")
    temperature: float = 0.8
    max_tokens: int = 2048
    max_retries: int = 3

    exec_timeout: int = 5

    ray_num_actors: int = 8
    ray_address: Optional[str] = None

    max_samples_per_task: int = 16
    max_mc_steps: int = 3

    save_rollouts: bool = False
    flush_every: int = 1
    seed: int = 42


class TaskEvaluator:
    def __init__(self, api_base_url: str, api_key: str, model_name: str, temperature: float, max_tokens: int, max_retries: int, exec_timeout: int):
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.exec_timeout = exec_timeout

    def extract_all_sections(self, text: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
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

    @staticmethod
    def normalize(s: Optional[str]) -> str:
        if not s:
            return ""
        return " ".join(s.split())

    def extract_sample_inputs(self, description: Optional[str]) -> List[str]:
        if not description:
            return []
        examples_match = re.search(r'(?:Example|Examples)\s*\n(.*)', description, re.DOTALL | re.IGNORECASE)
        target_text = examples_match.group(1) if examples_match else description
        raw_inputs = re.findall(r'Input\s*\n+(.*?)(?=\n+Output)', target_text, re.DOTALL | re.IGNORECASE)
        cleaned_inputs = []
        for raw in raw_inputs:
            clean_str = raw.replace("```text", "").replace("```", "").strip()
            if clean_str:
                cleaned_inputs.append(self.normalize(clean_str))
        return cleaned_inputs

    def run_code(self, language: str, code: str, input_data: str, timeout: Optional[int] = None) -> Tuple[str, str]:
        timeout = timeout or self.exec_timeout
        lang = (language or "").lower()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                if "c++" in lang or "cpp" in lang:
                    src_file = os.path.join(temp_dir, "solution.cpp")
                    exe_file = os.path.join(temp_dir, "solution.exe")
                    with open(src_file, "w", encoding="utf-8") as f:
                        f.write(code)
                    subprocess.check_call(["g++", "-O2", src_file, "-o", exe_file], stderr=subprocess.DEVNULL)
                    exec_cmd = [exe_file]
                elif "python" in lang:
                    src_file = os.path.join(temp_dir, "solution.py")
                    with open(src_file, "w", encoding="utf-8") as f:
                        f.write(code)
                    exec_cmd = ["python2" if "2" in lang else "python3", src_file]
                elif "java" in lang:
                    src_file = os.path.join(temp_dir, "Main.java")
                    with open(src_file, "w", encoding="utf-8") as f:
                        fixed_code = re.sub(r"public\s+class\s+\w+", "public class Main", code)
                        f.write(fixed_code)
                    subprocess.check_call(["javac", src_file], stderr=subprocess.DEVNULL)
                    exec_cmd = ["java", "-cp", temp_dir, "Main"]
                else:
                    return "UNSUPPORTED_LANG", ""

                process = subprocess.Popen(
                    exec_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="ignore",
                )
                stdout, stderr = process.communicate(input=input_data, timeout=timeout)
                if process.returncode != 0:
                    return "RUNTIME_ERROR", stderr
                return "SUCCESS", stdout.strip()
            except subprocess.CalledProcessError:
                return "COMPILE_ERROR", ""
            except subprocess.TimeoutExpired:
                process.kill()
                return "TIMEOUT", ""
            except Exception as e:
                return "SYSTEM_ERROR", str(e)

    def evaluate_reward(
        self,
        llm_output: str,
        oracle_lang: str,
        oracle_code: str,
        buggy_lang: str,
        buggy_code: str,
        cached_sample_inputs: List[str],
    ) -> Tuple[float, Dict[str, Any]]:
        sections = self.extract_all_sections(llm_output)
        detail: Dict[str, Any] = {
            "format_ok": False,
            "anti_copy_ok": False,
            "oracle_status": None,
            "oracle_output_match": False,
            "buggy_status": None,
            "bug_triggered": False,
            "normalized_input": "",
            "sections": sections,
            "failure_reason": None,
        }

        if not sections or not all(sections.values()):
            detail["failure_reason"] = "format_incomplete"
            return 0.0, detail
        detail["format_ok"] = True

        inp = sections["Input"] or ""
        expected_out = sections["Expected Output"] or ""
        normalized_inp = self.normalize(inp)
        detail["normalized_input"] = normalized_inp

        if normalized_inp in cached_sample_inputs:
            detail["failure_reason"] = "copied_example_input"
            return 0.0, detail
        detail["anti_copy_ok"] = True

        status, oracle_real_out = self.run_code(oracle_lang, oracle_code, inp)
        detail["oracle_status"] = status
        if status != "SUCCESS":
            detail["failure_reason"] = f"oracle_{status.lower()}"
            return 0.0, detail

        if self.normalize(oracle_real_out) != self.normalize(expected_out):
            detail["failure_reason"] = "expected_output_mismatch_oracle"
            return 0.0, detail
        detail["oracle_output_match"] = True

        b_status, b_out = self.run_code(buggy_lang, buggy_code, inp)
        detail["buggy_status"] = b_status

        if b_status in ["TIMEOUT", "RUNTIME_ERROR"]:
            detail["bug_triggered"] = True
            return 1.0, detail

        if b_status == "SUCCESS" and self.normalize(b_out) == self.normalize(oracle_real_out):
            detail["failure_reason"] = "buggy_matches_oracle"
            return 0.0, detail

        detail["bug_triggered"] = True
        return 1.0, detail

    def generate_completions(self, prompt: str, prefix: str = "", n: int = 16) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_header = ""
        if prefix:
            messages.append({"role": "assistant", "content": prefix})
            kwargs["extra_body"] = {
                "continue_final_message": True,
                "add_generation_prompt": False,
                "prefix": True,
            }
            last_header = prefix.strip().split("\n")[-1].strip()

        kwargs["messages"] = messages
        valid: List[str] = []

        for attempt in range(self.max_retries):
            needed = n - len(valid)
            if needed <= 0:
                break
            kwargs["n"] = needed
            try:
                response = self.client.chat.completions.create(**kwargs)
                for choice in response.choices:
                    comp = choice.message.content or ""
                    if last_header and last_header in comp:
                        continue
                    valid.append(comp)
                    if len(valid) == n:
                        break
            except Exception as e:
                print(f"[API warning] completion attempt {attempt + 1}/{self.max_retries} failed: {e}")
                time.sleep(2 ** attempt)

        while len(valid) < n:
            valid.append("")
        return valid[:n]


SECTION_ORDER = [
    "Bug Analysis",
    "Test Design",
    "Output Prediction",
    "Input",
    "Expected Output",
]

MC_SECTION_ORDER = [
    "Bug Analysis",
    "Test Design",
    "Output Prediction",
]


def build_prompt(task_row: Dict[str, Any], prompt_template: str) -> str:
    return prompt_template.format(
        description=task_row["description"],
        code=task_row["buggy_code"],
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


def difficulty_to_mc_k(base_score: Optional[float]) -> int:
    if base_score is None:
        return 64
    try:
        x = float(base_score)
    except Exception:
        return 64
    if 0.0 <= x < 0.1:
        return 64
    if 0.1 <= x < 0.9:
        return 48
    return 32


def sanitize_sections(sample: Dict[str, Any], evaluator: TaskEvaluator) -> Optional[Dict[str, str]]:
    sections = sample.get("sections")
    if not sections:
        raw = sample.get("raw_response") or ""
        sections = evaluator.extract_all_sections(raw)
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


def build_prefix_from_sections(sections: Dict[str, str], up_to_idx: int) -> str:
    blocks: List[str] = []
    for idx in range(up_to_idx + 1):
        header = MC_SECTION_ORDER[idx]
        blocks.append(f"### {header}\n{sections[header].strip()}")
    return "\n\n".join(blocks).rstrip() + "\n\n"


@ray.remote(num_cpus=1)
class Stage2Actor:
    def __init__(self, cfg_dict: Dict[str, Any], prompt_template: str):
        self.cfg = Stage2Config(**cfg_dict)
        self.prompt_template = prompt_template
        self.evaluator = TaskEvaluator(
            api_base_url=self.cfg.api_base_url,
            api_key=self.cfg.api_key,
            model_name=self.cfg.model_name,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            max_retries=self.cfg.max_retries,
            exec_timeout=self.cfg.exec_timeout,
        )
        self.rng = random.Random(self.cfg.seed)

    def process_task(self, task_row: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task_row.get("task_id", "unknown_task")
        prompt = build_prompt(task_row, self.prompt_template)
        cached_sample_inputs = self.evaluator.extract_sample_inputs(task_row.get("description", ""))
        mc_k = difficulty_to_mc_k(task_row.get("base_score"))

        all_rollout_rows: List[Dict[str, Any]] = []
        sample_rows: List[Dict[str, Any]] = []

        samples = task_row.get("samples", [])[: self.cfg.max_samples_per_task]
        num_valid_samples = 0
        num_total_mc_items = 0
        num_total_mc_scores = 0

        for sample_idx, sample in enumerate(samples):
            sample_record: Dict[str, Any] = {
                "sample_idx": sample_idx,
                "orig_label": sample.get("label"),
                "orig_reward": sample.get("reward"),
                "response_md5": sample.get("response_md5"),
                "raw_response": sample.get("raw_response"),
                "step_mc": [],
                "skipped": False,
                "skip_reason": None,
            }

            sections = sanitize_sections(sample, self.evaluator)
            if not sections:
                sample_record["skipped"] = True
                sample_record["skip_reason"] = "sections_invalid_or_incomplete"
                sample_rows.append(sample_record)
                continue

            num_valid_samples += 1
            sample_record["sections"] = sections

            for step_idx, step_name in enumerate(MC_SECTION_ORDER[: self.cfg.max_mc_steps]):
                prefix = build_prefix_from_sections(sections, step_idx)
                continuations = self.evaluator.generate_completions(prompt=prompt, prefix=prefix, n=mc_k)

                rollout_successes = 0
                rollout_rows_for_this_step: List[Dict[str, Any]] = []

                for rollout_idx, suffix in enumerate(continuations):
                    full_response = prefix + (suffix or "")
                    reward, detail = self.evaluator.evaluate_reward(
                        llm_output=full_response,
                        oracle_lang=task_row["oracle_lang"],
                        oracle_code=task_row["oracle_code"],
                        buggy_lang=task_row["buggy_lang"],
                        buggy_code=task_row["buggy_code"],
                        cached_sample_inputs=cached_sample_inputs,
                    )
                    success = int(reward > 0.5)
                    rollout_successes += success
                    num_total_mc_items += 1

                    if self.cfg.save_rollouts:
                        rollout_rows_for_this_step.append({
                            "task_id": task_id,
                            "sample_idx": sample_idx,
                            "step_idx": step_idx,
                            "step_name": step_name,
                            "rollout_idx": rollout_idx,
                            "reward": reward,
                            "label": success,
                            "prefix": prefix,
                            "continuation": suffix,
                            "full_response": full_response,
                            "eval_detail": {k: v for k, v in detail.items() if k != "sections"},
                            "sections": detail.get("sections"),
                        })

                mc_score = rollout_successes / mc_k if mc_k > 0 else 0.0
                num_total_mc_scores += 1
                step_record = {
                    "step_idx": step_idx,
                    "step_name": step_name,
                    "mc_k": mc_k,
                    "mc_successes": rollout_successes,
                    "mc_score": mc_score,
                    "prefix": prefix,
                }
                sample_record["step_mc"].append(step_record)
                if rollout_rows_for_this_step:
                    all_rollout_rows.extend(rollout_rows_for_this_step)

            sample_rows.append(sample_record)

        task_result = {
            "task_id": task_id,
            "description": task_row.get("description"),
            "oracle_code": task_row.get("oracle_code"),
            "oracle_lang": task_row.get("oracle_lang"),
            "buggy_code": task_row.get("buggy_code"),
            "buggy_lang": task_row.get("buggy_lang"),
            "base_score": task_row.get("base_score"),
            "retained_total": task_row.get("retained_total"),
            "retained_pos": task_row.get("retained_pos"),
            "retained_neg": task_row.get("retained_neg"),
            "max_samples_per_task": self.cfg.max_samples_per_task,
            "processed_samples": len(samples),
            "valid_samples": num_valid_samples,
            "mc_k": mc_k,
            "mc_steps_computed": min(self.cfg.max_mc_steps, len(MC_SECTION_ORDER)),
            "num_total_mc_items": num_total_mc_items,
            "num_total_mc_scores": num_total_mc_scores,
            "samples": sample_rows,
        }

        return {
            "task_result": task_result,
            "rollout_rows": all_rollout_rows if self.cfg.save_rollouts else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 MC scoring with Ray.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="stage1_balanced_tasks.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--api_base_url", type=str, default=os.getenv("GENERATION_API_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("GENERATION_API_KEY", ""))
    parser.add_argument("--model_name", type=str, default=os.getenv("GENERATION_MODEL_NAME", ""))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--exec_timeout", type=int, default=5)

    parser.add_argument("--ray_num_actors", type=int, default=8)
    parser.add_argument("--ray_address", type=str, default=None)

    parser.add_argument("--max_samples_per_task", type=int, default=16)
    parser.add_argument("--max_mc_steps", type=int, default=3)
    parser.add_argument("--save_rollouts", action="store_true")
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Stage2Config(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        exec_timeout=args.exec_timeout,
        ray_num_actors=args.ray_num_actors,
        ray_address=args.ray_address,
        max_samples_per_task=args.max_samples_per_task,
        max_mc_steps=args.max_mc_steps,
        save_rollouts=args.save_rollouts,
        flush_every=max(1, args.flush_every),
        seed=args.seed,
    )

    missing_api = [name for name in ["api_base_url", "api_key", "model_name"] if not getattr(cfg, name)]
    if missing_api:
        raise ValueError(f"Missing required generation API settings: {', '.join(missing_api)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_out_path = output_dir / "mc_tasks.jsonl"
    rollouts_out_path = output_dir / "mc_rollouts.jsonl"
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
    actors = [Stage2Actor.remote(asdict(cfg), PROJECT_PROMPT_TEMPLATE) for _ in range(cfg.ray_num_actors)]

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
    total_valid_samples = 0
    total_processed_samples = 0
    total_mc_items = 0
    total_mc_scores = 0
    total_rollout_rows = 0

    buffer_task_rows: List[Dict[str, Any]] = []
    buffer_rollout_rows: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(tasks), desc="Stage-2 MC")

    def flush_buffers(force: bool = False) -> None:
        nonlocal buffer_task_rows, buffer_rollout_rows
        if not force and len(buffer_task_rows) < cfg.flush_every:
            return
        append_jsonl(tasks_out_path, buffer_task_rows)
        buffer_task_rows = []
        if cfg.save_rollouts:
            append_jsonl(rollouts_out_path, buffer_rollout_rows)
            buffer_rollout_rows = []
        stats = {
            "input_jsonl": cfg.input_jsonl,
            "remaining_run_tasks": len(tasks),
            "already_existing_tasks": len(existing_task_ids),
            "processed_tasks_in_this_run": processed_tasks,
            "processed_tasks_total_output_estimate": len(existing_task_ids) + processed_tasks,
            "total_processed_samples": total_processed_samples,
            "total_valid_samples": total_valid_samples,
            "total_mc_items": total_mc_items,
            "total_mc_scores": total_mc_scores,
            "total_rollout_rows": total_rollout_rows if cfg.save_rollouts else None,
            "avg_valid_samples_per_task": (total_valid_samples / processed_tasks) if processed_tasks else 0.0,
            "avg_mc_items_per_task": (total_mc_items / processed_tasks) if processed_tasks else 0.0,
            "config": asdict(cfg),
            "output_files": {
                "tasks": str(tasks_out_path),
                "rollouts": str(rollouts_out_path) if cfg.save_rollouts else None,
            },
        }
        write_json(stats_path, stats)

    while in_flight:
        done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        done_ref = done_refs[0]
        actor = in_flight.pop(done_ref)
        result = ray.get(done_ref)

        task_result = result["task_result"]
        rollout_rows = result.get("rollout_rows") or []

        processed_tasks += 1
        total_processed_samples += int(task_result.get("processed_samples", 0))
        total_valid_samples += int(task_result.get("valid_samples", 0))
        total_mc_items += int(task_result.get("num_total_mc_items", 0))
        total_mc_scores += int(task_result.get("num_total_mc_scores", 0))
        total_rollout_rows += len(rollout_rows)

        buffer_task_rows.append(task_result)
        if cfg.save_rollouts and rollout_rows:
            buffer_rollout_rows.extend(rollout_rows)

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

    print("\nDone.")
    print(f"Tasks output: {tasks_out_path}")
    if cfg.save_rollouts:
        print(f"Rollouts output: {rollouts_out_path}")
    print(f"Stats: {stats_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
