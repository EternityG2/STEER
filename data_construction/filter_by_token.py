import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


def format_input(description: str, code: str) -> str:
    return f"Description:\n{description}\n\nSolution:\n{code}"


def filter_solution_map(
    tokenizer,
    description: str,
    languages: List[str],
    codes: List[str],
    max_tokens: int,
    keep_multiple: bool,
) -> Tuple[Dict[str, Any], int]:
    kept: Dict[str, Any] = {}
    removed = 0
    for lang, code in zip(languages, codes):
        text = format_input(description, code)
        if len(tokenizer.encode(text, add_special_tokens=True)) > max_tokens:
            removed += 1
            continue
        if keep_multiple:
            kept.setdefault(lang, []).append(code)
        elif lang not in kept:
            kept[lang] = code
    return kept, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter solutions by tokenizer length.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_tokens", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    total_count = 0
    kept_count = 0
    removed_solutions = 0

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="token-filter"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_count += 1
            description = item.get("description", "")

            valid_correct, removed = filter_solution_map(
                tokenizer,
                description,
                item.get("solutions", {}).get("language", []),
                item.get("solutions", {}).get("solution", []),
                args.max_tokens,
                keep_multiple=False,
            )
            removed_solutions += removed

            valid_incorrect, removed = filter_solution_map(
                tokenizer,
                description,
                item.get("incorrect_solutions", {}).get("language", []),
                item.get("incorrect_solutions", {}).get("bug_solution", []),
                args.max_tokens,
                keep_multiple=True,
            )
            removed_solutions += removed

            final_langs = sorted(set(valid_correct) & set(valid_incorrect))
            if not final_langs:
                continue

            item["solutions"] = {
                "language": final_langs,
                "solution": [valid_correct[lang] for lang in final_langs],
            }
            item["incorrect_solutions"] = {
                "language": [lang for lang in final_langs for _ in valid_incorrect[lang]],
                "bug_solution": [code for lang in final_langs for code in valid_incorrect[lang]],
            }

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"Kept {kept_count}/{total_count} rows; removed {removed_solutions} long solutions.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
