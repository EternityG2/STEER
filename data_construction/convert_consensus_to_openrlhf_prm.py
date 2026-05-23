#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert strict consensus data to the OpenRLHF PRM JSONL format."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer


PLACEHOLDER = "ки"
POS_TOKEN = "+"
NEG_TOKEN = "-"
from prompts import Prompt as PROMPT_TEMPLATE
 
REQUIRED_SECTION_KEYS = ["Bug Analysis", "Test Design", "Output Prediction"]


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"JSONL parse error at {path}:{line_no}: {e}")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def to_label_token(x: int) -> str:
    return POS_TOKEN if int(x) == 1 else NEG_TOKEN


def get_task_id(obj: Dict) -> Optional[str]:
    return obj.get("task_id") or obj.get("id") or obj.get("problem_id")


def get_samples(obj: Dict) -> List[Dict]:
    for key in ["samples", "balanced_samples", "kept_samples", "selected_samples"]:
        val = obj.get(key)
        if isinstance(val, list):
            return val
    return []


def build_prompt(description: str, buggy_code: str) -> str:
    return PROMPT_TEMPLATE.format(
        description=normalize_text(description),
        code=str(buggy_code).rstrip(),
    )


def build_candidate_prefix(sections: Dict[str, str], placeholder: str) -> str:
    bug_analysis = normalize_text(sections["Bug Analysis"])
    test_design = normalize_text(sections["Test Design"])
    output_prediction = normalize_text(sections["Output Prediction"])

    return (
        "### Bug Analysis\n"
        f"{bug_analysis}\n{placeholder}\n"
        "### Test Design\n"
        f"{test_design}\n{placeholder}\n"
        "### Output Prediction\n"
        f"{output_prediction}\n{placeholder}\n"
    )


def validate_sections(sections: Dict) -> Tuple[bool, str]:
    if not isinstance(sections, dict):
        return False, "sections_not_dict"
    for key in REQUIRED_SECTION_KEYS:
        if key not in sections:
            return False, f"missing_section_{key}"
        if not normalize_text(sections[key]):
            return False, f"empty_section_{key}"
    return True, "ok"


def get_consensus_labels(sample: Dict, prefer: str = "mc") -> Optional[List[int]]:
    if prefer == "judge":
        labels = sample.get("judge_labels_normalized")
        if isinstance(labels, list) and len(labels) >= 3:
            return [int(x) for x in labels[:3]]
    labels = sample.get("mc_labels_normalized")
    if isinstance(labels, list) and len(labels) >= 3:
        return [int(x) for x in labels[:3]]
    labels = sample.get("judge_labels_normalized")
    if isinstance(labels, list) and len(labels) >= 3:
        return [int(x) for x in labels[:3]]
    return None


def build_chat_input(
    tokenizer,
    description: str,
    buggy_code: str,
    sections: Dict[str, str],
    placeholder: str,
) -> str:
    prompt = build_prompt(description, buggy_code)
    candidate = build_candidate_prefix(sections, placeholder)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": candidate},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def convert_sample_to_row(
    tokenizer,
    description: str,
    buggy_code: str,
    sample: Dict,
    placeholder: str,
    label_source: str,
) -> Tuple[Optional[Dict], str]:
    if sample.get("strict_consensus_keep", True) is False:
        return None, "strict_consensus_keep_false"

    sections = sample.get("sections")
    ok, reason = validate_sections(sections)
    if not ok:
        return None, reason

    labels = get_consensus_labels(sample, prefer=label_source)
    if labels is None:
        return None, "missing_consensus_labels"
    if len(labels) != 3 or any(x not in (0, 1) for x in labels):
        return None, "invalid_consensus_labels"

    try:
        input_text = build_chat_input(
            tokenizer=tokenizer,
            description=description,
            buggy_code=buggy_code,
            sections=sections,
            placeholder=placeholder,
        )
    except Exception as e:
        return None, f"build_chat_input_error:{type(e).__name__}"

    label_tokens = [to_label_token(x) for x in labels]

    if input_text.count(placeholder) != len(label_tokens):
        return None, f"placeholder_mismatch_{input_text.count(placeholder)}_vs_{len(label_tokens)}"

    row = {
        "input": input_text,
        "label": label_tokens,
    }
    return row, "ok"


def convert_all(
    src_path: Path,
    tokenizer,
    placeholder: str,
    label_source: str,
) -> Tuple[List[Dict], Counter]:
    rows = []
    reject_stats = Counter()

    for obj in read_jsonl(src_path):
        task_id = get_task_id(obj)
        description = obj.get("description")
        buggy_code = obj.get("buggy_code")
        samples = get_samples(obj)

        if not task_id:
            reject_stats["missing_task_id"] += 1
            continue
        if not description:
            reject_stats["missing_description"] += 1
            continue
        if not buggy_code:
            reject_stats["missing_buggy_code"] += 1
            continue
        if not isinstance(samples, list) or len(samples) == 0:
            reject_stats["empty_samples"] += 1
            continue

        for sample in samples:
            row, reason = convert_sample_to_row(
                tokenizer=tokenizer,
                description=description,
                buggy_code=buggy_code,
                sample=sample,
                placeholder=placeholder,
                label_source=label_source,
            )
            if row is None:
                reject_stats[reason] += 1
                continue

            rows.append({
                "task_id": task_id,
                "row": row,
            })

    return rows, reject_stats


def split_by_task_with_target_test_rows(
    rows_with_tid: List[Dict],
    target_test_rows: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], Dict]:
    rng = random.Random(seed)

    grouped = defaultdict(list)
    for item in rows_with_tid:
        grouped[item["task_id"]].append(item["row"])

    task_ids = list(grouped.keys())
    rng.shuffle(task_ids)

    test_task_ids = []
    current_test_rows = 0

    for tid in task_ids:
        task_rows = len(grouped[tid])
        if current_test_rows >= target_test_rows:
            break
        test_task_ids.append(tid)
        current_test_rows += task_rows

    test_task_id_set = set(test_task_ids)

    train_rows = []
    test_rows = []
    for tid, items in grouped.items():
        if tid in test_task_id_set:
            test_rows.extend(items)
        else:
            train_rows.extend(items)

    split_info = {
        "num_total_tasks": len(task_ids),
        "num_test_tasks": len(test_task_id_set),
        "num_train_tasks": len(task_ids) - len(test_task_id_set),
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "target_test_rows": target_test_rows,
    }
    return train_rows, test_rows, split_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Input strict consensus retained jsonl.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Tokenizer/model path used for apply_chat_template.",
    )
    parser.add_argument(
        "--test_rows",
        type=int,
        default=500,
        help="Approximate number of test rows (split by task_id).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--placeholder",
        type=str,
        default=PLACEHOLDER,
        help="Placeholder token after each supervised step.",
    )
    parser.add_argument(
        "--label_source",
        type=str,
        default="mc",
        choices=["mc", "judge"],
        help="Which normalized consensus label field to use.",
    )
    args = parser.parse_args()

    src_path = Path(args.src_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template. Please use an instruct/chat tokenizer.")

    print("[2] Converting rows...")
    rows_with_tid, reject_stats = convert_all(
        src_path=src_path,
        tokenizer=tokenizer,
        placeholder=args.placeholder,
        label_source=args.label_source,
    )

    usable_rows = len(rows_with_tid)
    print(f"usable_rows={usable_rows}")

    if reject_stats:
        print("\n[Reject stats]")
        for k, v in reject_stats.most_common():
            print(f"{k}: {v}")

    if usable_rows == 0:
        write_jsonl(out_dir / "train.jsonl", [])
        write_jsonl(out_dir / "test.jsonl", [])
        stats = {
            "src_path": str(src_path),
            "usable_rows": 0,
            "reject_stats": dict(reject_stats),
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nNo usable rows. Empty files written to: {out_dir.resolve()}")
        return

    print("[3] Splitting by task_id...")
    train_rows, test_rows, split_info = split_by_task_with_target_test_rows(
        rows_with_tid=rows_with_tid,
        target_test_rows=args.test_rows,
        seed=args.seed,
    )

    print(f"train_rows={len(train_rows)}, test_rows={len(test_rows)}")

    print("[4] Writing files...")
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    stats = {
        "src_path": str(src_path),
        "out_dir": str(out_dir),
        "model_path": args.model_path,
        "label_source": args.label_source,
        "placeholder": args.placeholder,
        "usable_rows": usable_rows,
        "reject_stats": dict(reject_stats),
        "split_info": split_info,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. Files written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
