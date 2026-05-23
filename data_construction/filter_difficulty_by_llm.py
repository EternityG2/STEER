import argparse
import hashlib
import itertools
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


@dataclass
class Config:
    input_jsonl: str
    output_jsonl: str
    api_base_url: str
    api_key: str
    model_name: str
    num_generations: int
    max_workers: int
    batch_size: int
    debug_limit: Optional[int]
    timeout: int


def stable_id(*parts: Any) -> str:
    raw = "::".join(str(x) for x in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def extract_code(content: str) -> str:
    if "```python" in content:
        return content.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in content:
        return content.split("```", 1)[1].split("```", 1)[0].strip()
    return content.strip()


def get_llm_solution(client: OpenAI, cfg: Config, description: str) -> str:
    prompt = (
        "Please solve the following programming problem in Python 3. "
        "Provide only the code, wrapped in ```python ... ``` blocks.\n\n"
        f"Problem Description:\n{description}"
    )
    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return extract_code(response.choices[0].message.content or "")
    except Exception as exc:
        print(f"LLM request failed: {exc}")
        return ""


def run_test(code: str, input_str: str, expected_output: str, timeout: int) -> bool:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        process = subprocess.run(
            ["python3", tmp_path],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return process.stdout.strip() == expected_output.strip()
    except Exception:
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def normalize_test_cases(raw: Any) -> List[Tuple[str, str]]:
    if isinstance(raw, dict):
        inputs = raw.get("inputs") or raw.get("input") or []
        outputs = raw.get("outputs") or raw.get("output") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        return [(str(i), str(o)) for i, o in zip(inputs, outputs)]
    if isinstance(raw, list):
        pairs = []
        for case in raw:
            if isinstance(case, str):
                try:
                    case = json.loads(case)
                except json.JSONDecodeError:
                    continue
            if isinstance(case, dict):
                inp = case.get("input", case.get("inputs", case.get("in", "")))
                out = case.get("output", case.get("outputs", case.get("out", "")))
                pairs.append((str(inp), str(out)))
        return pairs
    return []


def get_public_tests(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    return normalize_test_cases(item.get("test_cases")) or normalize_test_cases(item.get("public_tests"))


def validate_problem(client: OpenAI, cfg: Config, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    io_pairs = get_public_tests(item)
    if not io_pairs:
        return None

    success_count = 0
    description = item.get("description", "")
    for _ in range(cfg.num_generations):
        generated_code = get_llm_solution(client, cfg, description)
        if not generated_code:
            continue
        if all(run_test(generated_code, inp, out, cfg.timeout) for inp, out in io_pairs):
            success_count += 1

    pass_rate = success_count / cfg.num_generations if cfg.num_generations else 0.0
    if pass_rate not in {0.5, 1.0}:
        return None

    item = dict(item)
    item["llm_pass_rate"] = pass_rate
    return item


def expand_to_training_tasks(item: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    correct_by_lang = dict(zip(item.get("solutions", {}).get("language", []), item.get("solutions", {}).get("solution", [])))
    inc_langs = item.get("incorrect_solutions", {}).get("language", [])
    inc_codes = item.get("incorrect_solutions", {}).get("bug_solution", [])
    source_id = item.get("id") or item.get("name") or stable_id(item.get("description", ""))

    for idx, (buggy_lang, buggy_code) in enumerate(zip(inc_langs, inc_codes)):
        oracle_code = correct_by_lang.get(buggy_lang)
        if not oracle_code:
            continue
        yield {
            "task_id": stable_id(source_id, buggy_lang, idx),
            "source_id": source_id,
            "description": item.get("description", ""),
            "oracle_code": oracle_code,
            "oracle_lang": buggy_lang,
            "buggy_code": buggy_code,
            "buggy_lang": buggy_lang,
            "base_score": item.get("llm_pass_rate"),
            "difficulty": item.get("difficulty"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter tasks by LLM solve rate and emit task-level rows.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--api_base_url", default=os.getenv("DIFFICULTY_API_BASE_URL", ""))
    parser.add_argument("--api_key", default=os.getenv("DIFFICULTY_API_KEY", ""))
    parser.add_argument("--model_name", default=os.getenv("DIFFICULTY_MODEL_NAME", ""))
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--debug_limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=2)
    return parser.parse_args()


def validate_config(cfg: Config) -> None:
    missing = [name for name in ["api_base_url", "api_key", "model_name"] if not getattr(cfg, name)]
    if missing:
        raise ValueError(f"Missing required API settings: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    cfg = Config(**vars(args))
    validate_config(cfg)

    input_path = Path(cfg.input_jsonl)
    output_path = Path(cfg.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.api_base_url)
    processed_count = 0
    saved_count = 0

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        file_iterator = itertools.islice(f_in, cfg.debug_limit) if cfg.debug_limit is not None else f_in
        total_items = cfg.debug_limit
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            with tqdm(total=total_items, desc="difficulty-filter") as pbar:
                while True:
                    lines_batch = list(itertools.islice(file_iterator, cfg.batch_size))
                    if not lines_batch:
                        break

                    items = []
                    for line in lines_batch:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            pbar.update(1)

                    futures = [executor.submit(validate_problem, client, cfg, item) for item in items]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            for task in expand_to_training_tasks(result):
                                f_out.write(json.dumps(task, ensure_ascii=False) + "\n")
                                saved_count += 1
                        processed_count += 1
                        pbar.update(1)

    print(f"Processed {processed_count} rows; saved {saved_count} task rows.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
