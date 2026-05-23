#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert stage2 MC scores into hard labels using Relative Progress Estimation (RPE).

Paper-aligned rule:
    P_t = MC(s_t, a_t) / MC(s_t)
    hard_label_t = 1 if P_t >= epsilon else 0

For this coding-task adaptation:
- step 1 uses task.base_score as MC(s1)
- step 2 uses step1.mc_score as denominator
- step 3 uses step2.mc_score as denominator

If a denominator <= 0 (or missing), the current step and all following steps are set to 0.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


STEP_NAMES = ["Bug Analysis", "Test Design", "Output Prediction"]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"JSONL parse error at {path}:{line_no}: {e}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_samples(task_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["samples", "balanced_samples", "kept_samples", "selected_samples"]:
        val = task_obj.get(key)
        if isinstance(val, list):
            return val
    return []


def set_samples(task_obj: Dict[str, Any], new_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(task_obj)
    for key in ["samples", "balanced_samples", "kept_samples", "selected_samples"]:
        if isinstance(out.get(key), list):
            out[key] = new_samples
            return out
    out["samples"] = new_samples
    return out


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def convert_one_sample(sample: Dict[str, Any], epsilon: float, task_base_score: Any) -> Dict[str, Any]:
    s = copy.deepcopy(sample)

    if s.get("skipped", False):
        s["rpe_skipped"] = True
        s["rpe_skip_reason"] = s.get("skip_reason", "upstream_skipped")
        s["step_hard_labels"] = []
        s["step_rpe"] = []
        return s

    step_mc = s.get("step_mc")
    if not isinstance(step_mc, list) or len(step_mc) == 0:
        s["rpe_skipped"] = True
        s["rpe_skip_reason"] = "missing_step_mc"
        s["step_hard_labels"] = []
        s["step_rpe"] = []
        return s

    step_mc = step_mc[:3]
    prev_state_score = to_float(task_base_score)

    step_hard_labels = []
    step_rpe = []
    dead = False

    for i, item in enumerate(step_mc):
        mc_score = to_float(item.get("mc_score"))
        step_name = item.get("step_name", STEP_NAMES[i] if i < len(STEP_NAMES) else f"step_{i}")

        info = {
            "step_idx": item.get("step_idx", i),
            "step_name": step_name,
            "mc_score": mc_score,
            "denominator_score": prev_state_score,
            "epsilon": epsilon,
            "rpe_ratio": None,
            "hard_label": 0,
            "rpe_reason": None,
        }

        if dead:
            info["rpe_reason"] = "previous_state_zero_or_invalid"
            step_hard_labels.append(0)
            step_rpe.append(info)
            prev_state_score = 0.0
            continue

        if mc_score is None:
            info["rpe_reason"] = "missing_mc_score"
            step_hard_labels.append(0)
            step_rpe.append(info)
            prev_state_score = 0.0
            dead = True
            continue

        if prev_state_score is None or prev_state_score <= 0:
            info["rpe_reason"] = "invalid_denominator"
            step_hard_labels.append(0)
            step_rpe.append(info)
            prev_state_score = 0.0
            dead = True
            continue

        ratio = mc_score / prev_state_score
        label = 1 if ratio >= epsilon else 0

        info["rpe_ratio"] = ratio
        info["hard_label"] = label
        info["rpe_reason"] = "ok"
        step_hard_labels.append(label)
        step_rpe.append(info)

        prev_state_score = mc_score
        if mc_score <= 0:
            dead = True

    s["rpe_skipped"] = False
    s["rpe_skip_reason"] = None
    s["step_hard_labels"] = step_hard_labels
    s["step_rpe"] = step_rpe
    s["rpe_first_incorrect_step"] = next((idx + 1 for idx, x in enumerate(step_hard_labels) if x == 0), -1)
    s["rpe_all_positive"] = all(x == 1 for x in step_hard_labels) if step_hard_labels else False
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="stage2_mc_tasks.jsonl")
    parser.add_argument("--output_jsonl", type=str, required=True, help="output jsonl")
    parser.add_argument("--stats_json", type=str, required=True, help="output stats json")
    parser.add_argument("--epsilon", type=float, default=0.8, help="RPE threshold epsilon")
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    stats_path = Path(args.stats_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    out_rows = []
    stats = {
        "input_jsonl": str(in_path),
        "output_jsonl": str(out_path),
        "epsilon": args.epsilon,
        "step1_denominator": "task.base_score",
        "tasks_total": 0,
        "samples_total": 0,
        "samples_rpe_skipped": 0,
        "samples_converted": 0,
        "step_items_total": 0,
        "hard_label_1_total": 0,
        "hard_label_0_total": 0,
        "first_incorrect_step_counts": {"-1": 0, "1": 0, "2": 0, "3": 0},
    }

    for task_obj in read_jsonl(in_path):
        stats["tasks_total"] += 1
        task_base_score = task_obj.get("base_score")
        samples = get_samples(task_obj)
        new_samples = []

        for sample in samples:
            stats["samples_total"] += 1
            new_sample = convert_one_sample(sample, args.epsilon, task_base_score)
            new_samples.append(new_sample)

            if new_sample.get("rpe_skipped", False):
                stats["samples_rpe_skipped"] += 1
            else:
                stats["samples_converted"] += 1
                labels = new_sample.get("step_hard_labels", [])
                stats["step_items_total"] += len(labels)
                stats["hard_label_1_total"] += sum(1 for x in labels if x == 1)
                stats["hard_label_0_total"] += sum(1 for x in labels if x == 0)
                fis = str(new_sample.get("rpe_first_incorrect_step", -1))
                if fis not in stats["first_incorrect_step_counts"]:
                    stats["first_incorrect_step_counts"][fis] = 0
                stats["first_incorrect_step_counts"][fis] += 1

        out_rows.append(set_samples(task_obj, new_samples))

    write_jsonl(out_path, out_rows)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
