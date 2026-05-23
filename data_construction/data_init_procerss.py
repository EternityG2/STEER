import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset
from tqdm import tqdm


LANG_MAP = {
    1: "Python 2",
    2: "C++",
    3: "Python 3",
    4: "Java",
}

KEEP_FIELDS = [
    "name",
    "description",
    "public_tests",
    "private_tests",
    "generated_tests",
    "time_limit",
    "memory_limit_bytes",
    "difficulty",
]


def parse_int_set(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def normalize_tests(raw_tests: Any) -> Dict[str, List[str]]:
    if not raw_tests:
        return {"inputs": [], "outputs": []}
    if isinstance(raw_tests, dict):
        inputs = raw_tests.get("inputs") or raw_tests.get("input") or []
        outputs = raw_tests.get("outputs") or raw_tests.get("output") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        return {"inputs": [str(x) for x in inputs], "outputs": [str(x) for x in outputs]}
    if isinstance(raw_tests, list):
        inputs, outputs = [], []
        for case in raw_tests:
            if isinstance(case, str):
                try:
                    case = json.loads(case)
                except json.JSONDecodeError:
                    continue
            if not isinstance(case, dict):
                continue
            inputs.append(str(case.get("input", case.get("inputs", case.get("in", "")))))
            outputs.append(str(case.get("output", case.get("outputs", case.get("out", "")))))
        return {"inputs": inputs, "outputs": outputs}
    return {"inputs": [], "outputs": []}


def collect_language_pairs(raw_solution: Dict[str, Any], raw_incorrect: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    sol_lang_ids = raw_solution.get("language", [])
    inc_lang_ids = raw_incorrect.get("language", [])
    valid_sol_langs = {lang for lang in sol_lang_ids if lang in LANG_MAP}
    valid_inc_langs = {lang for lang in inc_lang_ids if lang in LANG_MAP}
    shared_langs = valid_sol_langs & valid_inc_langs
    if not shared_langs:
        return [], [], [], []

    sol_langs, sol_codes, seen_correct = [], [], set()
    for lang_id, code in zip(sol_lang_ids, raw_solution.get("solution", [])):
        if lang_id in shared_langs and lang_id not in seen_correct:
            sol_langs.append(LANG_MAP[lang_id])
            sol_codes.append(code)
            seen_correct.add(lang_id)

    inc_langs, inc_codes = [], []
    for lang_id, code in zip(inc_lang_ids, raw_incorrect.get("solution", [])):
        if lang_id in shared_langs:
            inc_langs.append(LANG_MAP[lang_id])
            inc_codes.append(code)

    return sol_langs, sol_codes, inc_langs, inc_codes


def iter_filtered_rows(dataset: Iterable[Dict[str, Any]], target_difficulties: set[int]) -> Iterable[Dict[str, Any]]:
    for item in dataset:
        if (item.get("input_file") or "").strip() or (item.get("output_file") or "").strip():
            continue
        if item.get("difficulty") not in target_difficulties:
            continue

        raw_solution = item.get("solutions") or {}
        raw_incorrect = item.get("incorrect_solutions") or {}
        if not raw_solution or not raw_incorrect:
            continue

        sol_langs, sol_codes, inc_langs, inc_codes = collect_language_pairs(raw_solution, raw_incorrect)
        if not sol_langs or not inc_langs:
            continue

        row = {key: item[key] for key in KEEP_FIELDS if key in item}
        row["id"] = item.get("id") or item.get("name")
        row["test_cases"] = normalize_tests(item.get("public_tests"))
        row["solutions"] = {"language": sol_langs, "solution": sol_codes}
        row["incorrect_solutions"] = {"language": inc_langs, "bug_solution": inc_codes}
        yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize the Code-Contests-Plus dataset.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--dataset_name", default="ByteDance-Seed/Code-Contests-Plus")
    parser.add_argument("--dataset_config", default="3x")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--hf_endpoint", default=os.getenv("HF_ENDPOINT", ""))
    parser.add_argument("--target_difficulties", default="7,8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )[args.dataset_split]

    target_difficulties = parse_int_set(args.target_difficulties)
    saved_count = 0
    with output_path.open("w", encoding="utf-8") as f_out:
        for row in tqdm(iter_filtered_rows(dataset, target_difficulties), desc="data-init"):
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            saved_count += 1

    print(f"Saved {saved_count} rows to {output_path}")


if __name__ == "__main__":
    main()
