import argparse
import json
from pathlib import Path

from datasets import Dataset
from prompts import Prompt as PROMPT_TEMPLATE


def convert_jsonl_to_parquet(input_jsonl: str, output_parquet: str) -> None:
    rows = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if all(item.get(field) for field in ["description", "oracle_code", "oracle_lang", "buggy_code", "buggy_lang"]):
                task_items = [item]
            else:
                desc = item.get("description", "")
                sols = item.get("solutions", {})
                if not sols or not sols.get("solution"):
                    continue
                oracle_code = sols["solution"][0]
                oracle_lang = sols["language"][0]
                bad_sols = item.get("incorrect_solutions", {})
                task_items = []
                for idx, buggy_code in enumerate(bad_sols.get("bug_solution", [])):
                    buggy_langs = bad_sols.get("language", [])
                    task_items.append({
                        "description": desc,
                        "oracle_code": oracle_code,
                        "oracle_lang": oracle_lang,
                        "buggy_code": buggy_code,
                        "buggy_lang": buggy_langs[idx] if idx < len(buggy_langs) else "",
                    })

            for task in task_items:
                formatted_prompt = PROMPT_TEMPLATE.format(
                    description=task["description"],
                    code=task["buggy_code"],
                )
                rows.append({
                    "prompt": [{"role": "user", "content": formatted_prompt}],
                    "reward_model": {
                        "ground_truth": {
                            "description": task["description"],
                            "oracle_code": task["oracle_code"],
                            "oracle_lang": task["oracle_lang"],
                            "buggy_code": task["buggy_code"],
                            "buggy_lang": task["buggy_lang"],
                        }
                    },
                    "data_source": "code_contests_prm",
                })

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_path))
    print(f"Saved {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL training data to verl-compatible parquet.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_parquet", required=True)
    args = parser.parse_args()
    convert_jsonl_to_parquet(args.input_jsonl, args.output_parquet)


if __name__ == "__main__":
    main()
