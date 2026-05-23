import argparse
import json
import multiprocessing
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List

from openai import OpenAI
from prompts import Prompt


GLOBAL_EVALUATOR = None


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct generation evaluation.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--api_base_url", default=os.getenv("API_BASE_URL", ""))
    parser.add_argument("--api_key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME", ""))
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--run_timeout", type=int, default=3)
    parser.add_argument("--max_tasks", type=int, default=-1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    missing = [name for name in ["api_base_url", "api_key", "model_name"] if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required API settings: {', '.join(missing)}")


def classify_difficulty(score: Any) -> str:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if 0.0 <= score < 0.1:
        return "hard"
    if 0.1 <= score < 0.5:
        return "middle"
    if 0.5 <= score <= 1.0:
        return "easy"
    return "unknown"


def init_stats() -> Dict[str, int]:
    return {"total": 0, "pass_extr": 0, "pass_pc": 0, "pass_bt": 0, "pass_pc_bt": 0}


def update_stats(stats: Dict[str, int], res: Dict[str, Any]) -> None:
    stats["total"] += 1
    stats["pass_extr"] += res["pass_extr"]
    stats["pass_pc"] += res["pass_pc"]
    stats["pass_bt"] += res["pass_bt"]
    stats["pass_pc_bt"] += res["pass_pc_bt"]


def print_stats(title: str, stats: Dict[str, int], num_generations: int) -> None:
    total = stats["total"]
    extr_rate = (stats["pass_extr"] / total * 100) if total else 0
    pred_rate = (stats["pass_pc"] / total * 100) if total else 0
    attack_rate = (stats["pass_pc_bt"] / total * 100) if total else 0
    print(title)
    print(f"Total Tasks Evaluated:    {total}")
    print(f"Pass@{num_generations} Extraction:        {stats['pass_extr']} ({extr_rate:.2f}%)")
    print(f"Pass@{num_generations} IO Acc:            {stats['pass_pc']} ({pred_rate:.2f}%)")
    print(f"Pass@{num_generations} Attack Rate:       {stats['pass_pc_bt']} ({attack_rate:.2f}%)")


def is_task_level_item(item: Dict[str, Any]) -> bool:
    required_fields = ["description", "oracle_code", "oracle_lang", "buggy_code", "buggy_lang"]
    return all(item.get(field) for field in required_fields)


def load_tasks(data_path: str, max_tasks: int = -1) -> List[Dict[str, Any]]:
    tasks = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if is_task_level_item(item):
                task = dict(item)
                task.setdefault("id", item.get("task_id", f"task_{idx}"))
                task["difficulty_bucket"] = classify_difficulty(item.get("score", item.get("base_score")))
                tasks.append(task)
            else:
                desc = item.get("description", "")
                sols = item.get("solutions", {})
                if not sols or not sols.get("solution"):
                    continue
                oracle_code = sols["solution"][0]
                oracle_lang = sols["language"][0]
                bad_sols = item.get("incorrect_solutions", {})
                b_codes = bad_sols.get("bug_solution", [])
                b_langs = bad_sols.get("language", [])
                for i, b_code in enumerate(b_codes):
                    tasks.append({
                        "id": f"task_{idx}_bug_{i}",
                        "description": desc,
                        "oracle_code": oracle_code,
                        "oracle_lang": oracle_lang,
                        "buggy_code": b_code,
                        "buggy_lang": b_langs[i] if i < len(b_langs) else "",
                        "difficulty_bucket": "unknown",
                    })
                    if max_tasks > 0 and len(tasks) >= max_tasks:
                        return tasks

            if max_tasks > 0 and len(tasks) >= max_tasks:
                return tasks[:max_tasks]
    return tasks


class TaskEvaluator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg["api_base_url"], api_key=cfg["api_key"])

    def extract_xml(self, text: str):
        if not text:
            return None, None

        def clean(s):
            if not s:
                return ""
            return s.replace("`" * 3, "").replace("<code>", "").replace("</code>", "").strip()

        xml_in = re.search(r"<input>\s*(.*?)\s*</input>", text, re.DOTALL | re.IGNORECASE)
        xml_out = re.search(r"<expected_output>\s*(.*?)\s*</expected_output>", text, re.DOTALL | re.IGNORECASE)
        if xml_in and xml_out:
            return clean(xml_in.group(1)), clean(xml_out.group(1))

        bts = "`" * 3
        block_pattern = r"(?:" + bts + r"[\w]*|<code>)(.*?)(?:" + bts + r"|</code>)"
        in_pattern = r"(?:###|##|\*\*|Input:)\s*Input[:\s]*.*?" + block_pattern
        out_pattern = r"(?:###|##|\*\*|Output:)\s*(?:Expected\s+)?Output(?!\s+(?:Prediction|Derivation|Analysis|Strategy))[:\s]*.*?" + block_pattern
        in_chunk_match = re.search(in_pattern, text, re.DOTALL | re.IGNORECASE)
        out_chunk_match = re.search(out_pattern, text, re.DOTALL | re.IGNORECASE)
        if in_chunk_match and out_chunk_match:
            return clean(in_chunk_match.group(1)), clean(out_chunk_match.group(1))

        plain_in_pattern = r"(?:###|##|\*\*|Input:)\s*Input[:\s]*\n+(.*?)(?=\n\s*(?:###|##|\*\*|Output|Expected))"
        plain_out_pattern = r"(?:###|##|\*\*|Output:)\s*(?:Expected\s+)?Output(?!\s+(?:Prediction|Derivation|Analysis))[:\s]*\n+(.*)"
        plain_in = re.search(plain_in_pattern, text, re.DOTALL | re.IGNORECASE)
        plain_out = re.search(plain_out_pattern, text, re.DOTALL | re.IGNORECASE)
        if plain_in and plain_out:
            return clean(plain_in.group(1)), clean(plain_out.group(1))
        return None, None

    @staticmethod
    def normalize(s: str) -> str:
        if not s:
            return ""
        return "\n".join([line.rstrip() for line in s.strip().splitlines()])

    def run_code(self, language: str, code: str, input_data: str):
        lang = language.lower()
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
                    code = re.sub(r"public\s+class\s+\w+", "public class Main", code)
                    with open(src_file, "w", encoding="utf-8") as f:
                        f.write(code)
                    subprocess.check_call(["javac", src_file], stderr=subprocess.DEVNULL)
                    exec_cmd = ["java", "-cp", temp_dir, "Main"]
                else:
                    return "UNSUPPORTED_LANG", ""

                process = subprocess.Popen(exec_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore")
                stdout, stderr = process.communicate(input=input_data, timeout=self.cfg["run_timeout"])
                if process.returncode != 0:
                    return "RUNTIME_ERROR", stderr
                return "SUCCESS", stdout.strip()
            except subprocess.CalledProcessError:
                return "COMPILE_ERROR", ""
            except subprocess.TimeoutExpired:
                process.kill()
                return "TIMEOUT", ""
            except Exception as exc:
                return "SYSTEM_ERROR", str(exc)

    def process_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        description = task_info["description"]
        buggy_code = task_info["buggy_code"]
        buggy_lang = task_info["buggy_lang"]
        oracle_code = task_info["oracle_code"]
        oracle_lang = task_info["oracle_lang"]
        task_id = task_info.get("task_id", task_info.get("id", "unknown"))

        task_result = {
            "task_id": task_id,
            "difficulty_bucket": task_info.get("difficulty_bucket", "unknown"),
            "pass_extr": 0,
            "pass_pc": 0,
            "pass_bt": 0,
            "pass_pc_bt": 0,
        }

        try:
            response = self.client.chat.completions.create(
                model=self.cfg["model_name"],
                messages=[{"role": "user", "content": Prompt.format(description=description, code=buggy_code)}],
                temperature=self.cfg["temperature"],
                max_tokens=self.cfg["max_tokens"],
                n=self.cfg["num_generations"],
            )
        except Exception:
            return task_result

        for choice in response.choices:
            llm_text = choice.message.content
            inp, expected_out = self.extract_xml(llm_text)
            if inp is None or expected_out is None:
                continue

            task_result["pass_extr"] = 1
            status, oracle_real_out = self.run_code(oracle_lang, oracle_code, inp)
            if status != "SUCCESS":
                continue

            is_pred_correct = self.normalize(oracle_real_out) == self.normalize(expected_out)
            if is_pred_correct:
                task_result["pass_pc"] = 1

            b_status, b_out = self.run_code(buggy_lang, buggy_code, inp)
            triggered = b_status in ["TIMEOUT", "RUNTIME_ERROR"] or (b_status == "SUCCESS" and self.normalize(b_out) != self.normalize(oracle_real_out))
            if triggered:
                task_result["pass_bt"] = 1
                if is_pred_correct:
                    task_result["pass_pc_bt"] = 1
        return task_result


def init_worker(cfg: Dict[str, Any]) -> None:
    global GLOBAL_EVALUATOR
    GLOBAL_EVALUATOR = TaskEvaluator(cfg)


def worker_entry(task: Dict[str, Any]) -> Dict[str, Any]:
    return GLOBAL_EVALUATOR.process_task(task)


def main() -> None:
    args = build_args()
    validate_args(args)
    cfg = vars(args)

    print(f"Loading data from {args.data_path}")
    tasks = load_tasks(args.data_path, max_tasks=args.max_tasks)
    print(f"Loaded {len(tasks)} tasks.")

    overall_stats = init_stats()
    difficulty_stats = {bucket: init_stats() for bucket in ["hard", "middle", "easy", "unknown"]}
    completed = 0
    total_tasks = len(tasks)

    with multiprocessing.Pool(processes=args.num_workers, initializer=init_worker, initargs=(cfg,)) as pool:
        for res in pool.imap_unordered(worker_entry, tasks):
            completed += 1
            update_stats(overall_stats, res)
            bucket = res.get("difficulty_bucket", "unknown")
            update_stats(difficulty_stats.get(bucket, difficulty_stats["unknown"]), res)
            if completed % 100 == 0:
                print(
                    f"Progress: {completed}/{total_tasks} | "
                    f"Extr: {overall_stats['pass_extr']} | IO Acc: {overall_stats['pass_pc']} | "
                    f"Attack Rate: {overall_stats['pass_pc_bt']}"
                )

    print("\nFINAL RESULTS")
    print_stats("Overall", overall_stats, args.num_generations)
    print_stats("Hard   [0.0, 0.1)", difficulty_stats["hard"], args.num_generations)
    print_stats("Middle [0.1, 0.5)", difficulty_stats["middle"], args.num_generations)
    print_stats("Easy   [0.5, 1.0]", difficulty_stats["easy"], args.num_generations)
    if difficulty_stats["unknown"]["total"] > 0:
        print_stats("Unknown", difficulty_stats["unknown"], args.num_generations)


if __name__ == "__main__":
    main()
