#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict consensus filtering between:
1) MC-guided hard labels (stage2_mc_hard_labels.jsonl)
2) LLM-as-a-judge labels (stage3_judge_tasks.jsonl)

Strict paper-style rule:
- First normalize labels so that after the first 0, all later labels become 0.
- Keep a sample ONLY IF normalized MC labels == normalized Judge labels.
- If there is any mismatch, discard the entire sample.

This is stricter than only matching first error location.

Outputs:
- strict_consensus_retained.jsonl       : retained samples only, empty tasks removed
- strict_consensus_all_with_flags.jsonl : all samples with keep/discard flags
- strict_consensus_stats.json           : summary statistics
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def get_task_id(obj: Dict[str, Any]) -> str:
    return obj.get("task_id") or obj.get("id") or obj.get("problem_id") or "UNKNOWN_TASK"


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


def sample_key(sample: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    return (
        sample.get("sample_idx", sample.get("attempt_idx")),
        sample.get("response_md5"),
        sample.get("raw_response"),
    )


def normalize_binary_labels(labels: Any) -> Optional[List[int]]:
    if not isinstance(labels, list) or len(labels) == 0:
        return None
    out = []
    for v in labels[:3]:
        if v not in (0, 1):
            return None
        out.append(int(v))
    return out


def earliest_error_normalize(labels: List[int]) -> List[int]:
    """
    After first 0, all later labels become 0.
    Examples:
      [1,0,1] -> [1,0,0]
      [0,1,1] -> [0,0,0]
      [1,1,1] -> [1,1,1]
    """
    out = []
    seen_zero = False
    for v in labels:
        if seen_zero:
            out.append(0)
        else:
            out.append(v)
            if v == 0:
                seen_zero = True
    return out


def first_incorrect(labels: Optional[List[int]]) -> Optional[int]:
    if not labels:
        return None
    for i, v in enumerate(labels):
        if v == 0:
            return i + 1
    return -1


def get_mc_labels(sample: Dict[str, Any]) -> Optional[List[int]]:
    labels = normalize_binary_labels(sample.get("step_hard_labels"))
    if labels is not None:
        return labels

    step_rpe = sample.get("step_rpe")
    if isinstance(step_rpe, list) and len(step_rpe) > 0:
        out = []
        for item in step_rpe[:3]:
            if not isinstance(item, dict):
                return None
            v = item.get("hard_label")
            if v not in (0, 1):
                return None
            out.append(int(v))
        return out
    return None


def get_judge_labels(sample: Dict[str, Any]) -> Optional[List[int]]:
    step_items = sample.get("judge_steps")
    if isinstance(step_items, list) and len(step_items) > 0:
        out = []
        for item in step_items[:3]:
            if not isinstance(item, dict):
                return None
            v = item.get("judge_label")
            if v not in (0, 1):
                return None
            out.append(int(v))
        return out
    return None


def build_task_index(path: Path) -> Dict[str, Dict[Tuple[Any, Any, Any], Dict[str, Any]]]:
    out: Dict[str, Dict[Tuple[Any, Any, Any], Dict[str, Any]]] = {}
    for task_obj in read_jsonl(path):
        tid = get_task_id(task_obj)
        smap: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
        for s in get_samples(task_obj):
            smap[sample_key(s)] = s
        out[tid] = smap
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_jsonl", type=str, required=True,
                        help="stage2_mc_hard_labels.jsonl")
    parser.add_argument("--judge_jsonl", type=str, required=True,
                        help="stage3_judge_tasks.jsonl")
    parser.add_argument("--output_retained_jsonl", type=str, required=True)
    parser.add_argument("--output_all_jsonl", type=str, required=True)
    parser.add_argument("--stats_json", type=str, required=True)
    args = parser.parse_args()

    mc_path = Path(args.mc_jsonl)
    judge_path = Path(args.judge_jsonl)
    retained_path = Path(args.output_retained_jsonl)
    all_path = Path(args.output_all_jsonl)
    stats_path = Path(args.stats_json)

    retained_path.parent.mkdir(parents=True, exist_ok=True)
    all_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    judge_index = build_task_index(judge_path)

    retained_rows = []
    all_rows = []

    stats = Counter()
    mismatch_counter = Counter()

    for mc_task_obj in read_jsonl(mc_path):
        tid = get_task_id(mc_task_obj)
        mc_samples = get_samples(mc_task_obj)
        judge_samples_map = judge_index.get(tid, {})

        retained_samples = []
        all_samples = []

        stats["tasks_total"] += 1
        if tid in judge_index:
            stats["tasks_found_in_both"] += 1
        else:
            stats["tasks_missing_in_judge"] += 1

        for mc_sample in mc_samples:
            stats["samples_total"] += 1
            merged = copy.deepcopy(mc_sample)
            key = sample_key(mc_sample)
            judge_sample = judge_samples_map.get(key)

            merged["strict_consensus_keep"] = False
            merged["strict_consensus_reason"] = None
            merged["judge_available"] = judge_sample is not None

            if judge_sample is None:
                merged["strict_consensus_reason"] = "missing_matching_judge_sample"
                mismatch_counter["missing_matching_judge_sample"] += 1
                all_samples.append(merged)
                continue

            merged["judge_steps"] = judge_sample.get("judge_steps")
            merged["judge_first_error_step"] = judge_sample.get("judge_first_error_step")
            merged["judge_all_correct"] = judge_sample.get("judge_all_correct")
            merged["judge_parse_ok"] = judge_sample.get("judge_parse_ok")
            merged["judge_repaired"] = judge_sample.get("judge_repaired")
            merged["judge_conclusion"] = judge_sample.get("judge_conclusion")
            merged["judge_raw_response"] = judge_sample.get("judge_raw_response")
            merged["judge_skipped"] = judge_sample.get("skipped", False)
            merged["judge_skip_reason"] = judge_sample.get("skip_reason")

            if mc_sample.get("rpe_skipped", False):
                merged["strict_consensus_reason"] = "mc_skipped"
                mismatch_counter["mc_skipped"] += 1
                all_samples.append(merged)
                continue

            if judge_sample.get("skipped", False):
                merged["strict_consensus_reason"] = "judge_skipped"
                mismatch_counter["judge_skipped"] += 1
                all_samples.append(merged)
                continue

            mc_labels_raw = get_mc_labels(mc_sample)
            judge_labels_raw = get_judge_labels(judge_sample)

            if mc_labels_raw is None:
                merged["strict_consensus_reason"] = "mc_labels_missing_or_invalid"
                mismatch_counter["mc_labels_missing_or_invalid"] += 1
                all_samples.append(merged)
                continue

            if judge_labels_raw is None:
                merged["strict_consensus_reason"] = "judge_labels_missing_or_invalid"
                mismatch_counter["judge_labels_missing_or_invalid"] += 1
                all_samples.append(merged)
                continue

            mc_labels_norm = earliest_error_normalize(mc_labels_raw)
            judge_labels_norm = earliest_error_normalize(judge_labels_raw)

            merged["mc_labels_raw"] = mc_labels_raw
            merged["judge_labels_raw"] = judge_labels_raw
            merged["mc_labels_normalized"] = mc_labels_norm
            merged["judge_labels_normalized"] = judge_labels_norm
            merged["mc_first_incorrect_step_normalized"] = first_incorrect(mc_labels_norm)
            merged["judge_first_incorrect_step_normalized"] = first_incorrect(judge_labels_norm)

            keep = (mc_labels_norm == judge_labels_norm)
            merged["strict_consensus_keep"] = keep
            merged["strict_consensus_reason"] = "normalized_exact_match" if keep else "normalized_exact_mismatch"

            if keep:
                stats["samples_retained"] += 1
                retained_samples.append(merged)
            else:
                mismatch_counter["normalized_exact_mismatch"] += 1

            all_samples.append(merged)

        if len(retained_samples) > 0:
            retained_rows.append(set_samples(mc_task_obj, retained_samples))
            stats["tasks_retained_nonempty"] += 1
        else:
            stats["tasks_retained_empty"] += 1

        all_rows.append(set_samples(mc_task_obj, all_samples))

    write_jsonl(retained_path, retained_rows)
    write_jsonl(all_path, all_rows)

    result = {
        "mc_jsonl": str(mc_path),
        "judge_jsonl": str(judge_path),
        "output_retained_jsonl": str(retained_path),
        "output_all_jsonl": str(all_path),
        "consensus_mode": "strict_normalized_exact_match",
        "summary": {
            "tasks_total": stats["tasks_total"],
            "tasks_found_in_both": stats["tasks_found_in_both"],
            "tasks_missing_in_judge": stats["tasks_missing_in_judge"],
            "tasks_retained_nonempty": stats["tasks_retained_nonempty"],
            "tasks_retained_empty": stats["tasks_retained_empty"],
            "samples_total": stats["samples_total"],
            "samples_retained": stats["samples_retained"],
            "retention_rate": (stats["samples_retained"] / stats["samples_total"]) if stats["samples_total"] else 0.0,
        },
        "mismatch_reason_counts": dict(mismatch_counter),
    }

    stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
