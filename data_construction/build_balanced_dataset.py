#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build balanced stage-1 trajectories for unit-test generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from tqdm import tqdm
from openai import OpenAI


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
class SamplerConfig:
    api_base_url: str = os.getenv("GENERATION_API_BASE_URL", "")
    api_key: str = os.getenv("GENERATION_API_KEY", "")
    model_name: str = os.getenv("GENERATION_MODEL_NAME", "")
    temperature: float = 0.8
    max_tokens: int = 2048
    max_retries: int = 3

    exec_timeout: int = 5

    batch_schedule: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])

    target_each: int = 8

    shuffle_selected: bool = True

    ray_num_actors: int = max(1, os.cpu_count() or 1)
    ray_address: Optional[str] = None

    save_all_attempts: bool = False
    seed: int = 42


class TaskEvaluator:
    def __init__(self, cfg: SamplerConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.api_base_url, api_key=cfg.api_key)

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
        timeout = timeout or self.cfg.exec_timeout
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
            detail["failure_reason"] = "bad_format"
            return 0.0, detail
        detail["format_ok"] = True

        inp = sections["Input"] or ""
        expected_out = sections["Expected Output"] or ""
        normalized_inp = self.normalize(inp)
        detail["normalized_input"] = normalized_inp

        if normalized_inp in cached_sample_inputs:
            detail["failure_reason"] = "copied_sample_input"
            return 0.0, detail
        detail["anti_copy_ok"] = True

        status, oracle_real_out = self.run_code(oracle_lang, oracle_code, inp)
        detail["oracle_status"] = status
        if status != "SUCCESS":
            detail["failure_reason"] = f"oracle_{status.lower()}"
            return 0.0, detail

        if self.normalize(oracle_real_out) != self.normalize(expected_out):
            detail["failure_reason"] = "expected_output_mismatch_with_oracle"
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
            "model": self.cfg.model_name,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
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
        valid_completions: List[str] = []

        for attempt in range(self.cfg.max_retries):
            needed = n - len(valid_completions)
            if needed <= 0:
                break
            kwargs["n"] = needed
            try:
                response = self.client.chat.completions.create(**kwargs)
                for choice in response.choices:
                    comp = choice.message.content or ""
                    if last_header and last_header in comp:
                        continue
                    valid_completions.append(comp)
                    if len(valid_completions) == n:
                        break
            except Exception as e:
                print(f"[API warning] completion attempt {attempt + 1}/{self.cfg.max_retries} failed: {e}")
                time.sleep(2 ** attempt)

        while len(valid_completions) < n:
            valid_completions.append("")
        return valid_completions[:n]


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_prompt(task: Dict[str, Any], prompt_template: str) -> str:
    return prompt_template.format(
        description=task["description"],
        code=task["buggy_code"],
    )


def make_attempt_record(task: Dict[str, Any], idx: int, raw_text: str, reward: float, detail: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.get("id"),
        "attempt_idx": idx,
        "reward": reward,
        "label": int(reward > 0.5),
        "raw_response": raw_text,
        "sections": detail.get("sections"),
        "eval_detail": {k: v for k, v in detail.items() if k != "sections"},
        "response_md5": stable_hash(raw_text or ""),
    }


@ray.remote(num_cpus=1)
class TaskSamplerActor:
    def __init__(self, cfg_dict: Dict[str, Any], prompt_template: str):
        self.cfg = SamplerConfig(**cfg_dict)
        self.prompt_template = prompt_template
        self.evaluator = TaskEvaluator(self.cfg)
        self.rng = random.Random(self.cfg.seed)

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task.get("id", "unknown_task")
        prompt = build_prompt(task, self.prompt_template)
        cached_sample_inputs = self.evaluator.extract_sample_inputs(task.get("description", ""))

        pos_attempts: List[Dict[str, Any]] = []
        neg_attempts: List[Dict[str, Any]] = []
        all_attempts: List[Dict[str, Any]] = []

        sampled = 0
        stop_reason = "reached_max_budget"

        for cumulative_target in self.cfg.batch_schedule:
            if cumulative_target <= sampled:
                continue
            delta = cumulative_target - sampled
            completions = self.evaluator.generate_completions(prompt=prompt, n=delta)

            for raw_text in completions:
                reward, detail = self.evaluator.evaluate_reward(
                    llm_output=raw_text,
                    oracle_lang=task["oracle_lang"],
                    oracle_code=task["oracle_code"],
                    buggy_lang=task["buggy_lang"],
                    buggy_code=task["buggy_code"],
                    cached_sample_inputs=cached_sample_inputs,
                )
                attempt = make_attempt_record(task, sampled, raw_text, reward, detail)
                sampled += 1
                all_attempts.append(attempt)
                if reward > 0.5:
                    pos_attempts.append(attempt)
                else:
                    neg_attempts.append(attempt)

            if len(pos_attempts) >= self.cfg.target_each and len(neg_attempts) >= self.cfg.target_each:
                stop_reason = "enough_balanced_samples"
                break

        if len(pos_attempts) == 0 or len(neg_attempts) == 0:
            return {
                "task_id": task_id,
                "dropped": True,
                "drop_reason": "single_sided_after_max_budget",
                "sampled_total": sampled,
                "num_pos": len(pos_attempts),
                "num_neg": len(neg_attempts),
                "stop_reason": stop_reason,
                "balanced_record": None,
                "all_attempts": all_attempts if self.cfg.save_all_attempts else None,
            }

        k = min(self.cfg.target_each, len(pos_attempts), len(neg_attempts))
        selected_pos = pos_attempts[:k]
        selected_neg = neg_attempts[:k]
        selected = selected_pos + selected_neg

        if self.cfg.shuffle_selected:
            local_rng = random.Random(f"{self.cfg.seed}:{task_id}")
            local_rng.shuffle(selected)

        balanced_record = {
            "task_id": task_id,
            "description": task.get("description"),
            "oracle_code": task.get("oracle_code"),
            "oracle_lang": task.get("oracle_lang"),
            "buggy_code": task.get("buggy_code"),
            "buggy_lang": task.get("buggy_lang"),
            "base_score": task.get("base_score"),
            "sampled_total": sampled,
            "num_pos_total": len(pos_attempts),
            "num_neg_total": len(neg_attempts),
            "retained_pos": k,
            "retained_neg": k,
            "retained_total": 2 * k,
            "stop_reason": stop_reason,
            "schedule": self.cfg.batch_schedule,
            "samples": selected,
        }

        return {
            "task_id": task_id,
            "dropped": False,
            "drop_reason": None,
            "sampled_total": sampled,
            "num_pos": len(pos_attempts),
            "num_neg": len(neg_attempts),
            "stop_reason": stop_reason,
            "balanced_record": balanced_record,
            "all_attempts": all_attempts if self.cfg.save_all_attempts else None,
        }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error: line={line_no}, err={e}") from e
            tasks.append(obj)
    return tasks


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced stage-1 dataset with Ray.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--api_base_url", type=str, default=os.getenv("GENERATION_API_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("GENERATION_API_KEY", ""))
    parser.add_argument("--model_name", type=str, default=os.getenv("GENERATION_MODEL_NAME", ""))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--exec_timeout", type=int, default=5)

    parser.add_argument("--target_each", type=int, default=8, help="Target retained positives and negatives per task.")
    parser.add_argument(
        "--batch_schedule",
        type=str,
        default="16,32,64,128,256",
        help="Cumulative sampling schedule, for example 16,32,64,128,256.",
    )
    parser.add_argument("--save_all_attempts", action="store_true")
    parser.add_argument("--no_shuffle_selected", action="store_true")

    parser.add_argument("--ray_num_actors", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--ray_address", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule = [int(x.strip()) for x in args.batch_schedule.split(",") if x.strip()]
    if not schedule:
        raise ValueError("batch_schedule must not be empty")
    if sorted(schedule) != schedule:
        raise ValueError("batch_schedule must be sorted in ascending order")

    cfg = SamplerConfig(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        exec_timeout=args.exec_timeout,
        batch_schedule=schedule,
        target_each=args.target_each,
        shuffle_selected=not args.no_shuffle_selected,
        ray_num_actors=args.ray_num_actors,
        ray_address=args.ray_address,
        save_all_attempts=args.save_all_attempts,
        seed=args.seed,
    )

    missing_api = [name for name in ["api_base_url", "api_key", "model_name"] if not getattr(cfg, name)]
    if missing_api:
        raise ValueError(f"Missing required generation API settings: {', '.join(missing_api)}")

    tasks = load_jsonl(args.input_jsonl)
    if not tasks:
        raise ValueError("input data is empty")

    ray.init(address=cfg.ray_address, ignore_reinit_error=True)

    actors = [
        TaskSamplerActor.remote(asdict(cfg), PROJECT_PROMPT_TEMPLATE)
        for _ in range(cfg.ray_num_actors)
    ]

    in_flight = {}
    balanced_rows: List[Dict[str, Any]] = []
    all_attempt_rows: List[Dict[str, Any]] = []

    task_iter = iter(tasks)
    for actor in actors:
        try:
            task = next(task_iter)
        except StopIteration:
            break
        ref = actor.process_task.remote(task)
        in_flight[ref] = actor

    pbar = tqdm(total=len(tasks), desc="Stage-1 sampling")

    dropped_single_sided = 0
    kept_tasks = 0
    total_retained_samples = 0
    total_sampled_attempts = 0

    while in_flight:
        done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        done_ref = done_refs[0]
        actor = in_flight.pop(done_ref)
        result = ray.get(done_ref)

        total_sampled_attempts += int(result["sampled_total"])

        if result["dropped"]:
            dropped_single_sided += 1
        else:
            kept_tasks += 1
            balanced_record = result["balanced_record"]
            if balanced_record is not None:
                balanced_rows.append(balanced_record)
                total_retained_samples += int(balanced_record["retained_total"])

        if cfg.save_all_attempts and result.get("all_attempts"):
            all_attempt_rows.extend(result["all_attempts"])

        pbar.update(1)

        try:
            next_task = next(task_iter)
            new_ref = actor.process_task.remote(next_task)
            in_flight[new_ref] = actor
        except StopIteration:
            pass

    pbar.close()

    balanced_path = output_dir / "balanced_tasks.jsonl"
    dump_jsonl(str(balanced_path), balanced_rows)

    all_attempts_path = output_dir / "all_attempts.jsonl"
    if cfg.save_all_attempts:
        dump_jsonl(str(all_attempts_path), all_attempt_rows)

    stats = {
        "input_jsonl": args.input_jsonl,
        "num_tasks_total": len(tasks),
        "num_tasks_kept": kept_tasks,
        "num_tasks_dropped_single_sided": dropped_single_sided,
        "keep_rate": kept_tasks / len(tasks),
        "total_sampled_attempts": total_sampled_attempts,
        "avg_sampled_attempts_per_task": total_sampled_attempts / len(tasks),
        "total_retained_samples": total_retained_samples,
        "avg_retained_samples_per_kept_task": (total_retained_samples / kept_tasks) if kept_tasks else 0.0,
        "config": asdict(cfg),
        "output_files": {
            "balanced_tasks": str(balanced_path),
            "all_attempts": str(all_attempts_path) if cfg.save_all_attempts else None,
        },
    }

    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Kept tasks: {kept_tasks}/{len(tasks)}")
    print(f"Dropped single-sided tasks: {dropped_single_sided}")
    print(f"Balanced dataset: {balanced_path}")
    if cfg.save_all_attempts:
        print(f"All attempts: {all_attempts_path}")
    print(f"Stats: {output_dir / 'stage1_stats.json'}")

    ray.shutdown()


if __name__ == "__main__":
    main()
