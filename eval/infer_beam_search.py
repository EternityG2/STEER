import json
import os

import re
import time
import argparse
import subprocess
import tempfile
import multiprocessing
import requests
from openai import OpenAI
from prompts import Prompt
from tqdm import tqdm



GLOBAL_EVALUATOR = None


def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_base_url", type=str, default=os.getenv("API_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", ""))
    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", ""))

    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--b1", type=int, required=True, help="Number of beams retained after each step.")
    parser.add_argument("--b2", type=int, required=True, help="Number of candidates sampled per beam at each step.")

    parser.add_argument("--prm_server_url", type=str, default=os.getenv("PRM_SERVER_URL", ""))
    parser.add_argument("--prm_timeout", type=float, default=60.0)
    parser.add_argument("--prm_placeholder", type=str, default="ки")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--final_temperature", type=float, default=0.0)
    parser.add_argument("--step_max_tokens", type=int, default=512)
    parser.add_argument("--final_max_tokens", type=int, default=1024)

    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--run_timeout", type=int, default=3)

    parser.add_argument(
        "--max_tasks",
        type=int,
        default=-1,
        help="Run only the first N tasks; -1 runs all tasks.",
    )

    args = parser.parse_args()
    missing = [name for name in ["api_base_url", "api_key", "model_name", "prm_server_url"] if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")
    return args



class TaskEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg["api_base_url"],
            api_key=cfg["api_key"],
        )

    def extract_xml(self, text):
        if not text:
            return None, None

        def clean(s):
            if not s:
                return ""
            return (
                s.replace("`" * 3, "")
                .replace("<code>", "")
                .replace("</code>", "")
                .strip()
            )

        xml_in = re.search(
            r"<input>\s*(.*?)\s*</input>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        xml_out = re.search(
            r"<expected_output>\s*(.*?)\s*</expected_output>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if xml_in and xml_out:
            return clean(xml_in.group(1)), clean(xml_out.group(1))

        bts = "`" * 3
        block_pattern = r"(?:" + bts + r"[\w]*|<code>)(.*?)(?:" + bts + r"|</code>)"
        in_pattern = r"(?:###|##|\*\*|Input:)\s*Input[:\s]*.*?" + block_pattern
        out_pattern = (
            r"(?:###|##|\*\*|Output:)\s*(?:Expected\s+)?"
            r"Output(?!\s+(?:Prediction|Derivation|Analysis|Strategy))[:\s]*.*?"
            + block_pattern
        )

        in_chunk_match = re.search(in_pattern, text, re.DOTALL | re.IGNORECASE)
        out_chunk_match = re.search(out_pattern, text, re.DOTALL | re.IGNORECASE)
        if in_chunk_match and out_chunk_match:
            return clean(in_chunk_match.group(1)), clean(out_chunk_match.group(1))

        plain_in_pattern = (
            r"(?:###|##|\*\*|Input:)\s*Input[:\s]*\n+"
            r"(.*?)(?=\n\s*(?:###|##|\*\*|Output|Expected))"
        )
        plain_out_pattern = (
            r"(?:###|##|\*\*|Output:)\s*(?:Expected\s+)?"
            r"Output(?!\s+(?:Prediction|Derivation|Analysis))[:\s]*\n+(.*)"
        )
        plain_in = re.search(plain_in_pattern, text, re.DOTALL | re.IGNORECASE)
        plain_out = re.search(plain_out_pattern, text, re.DOTALL | re.IGNORECASE)
        if plain_in and plain_out:
            return clean(plain_in.group(1)), clean(plain_out.group(1))

        return None, None

    def normalize(self, s):
        if not s:
            return ""
        return "\n".join([line.rstrip() for line in s.strip().splitlines()])

    def run_code(self, language, code, input_data, timeout=None):
        if timeout is None:
            timeout = self.cfg["run_timeout"]

        lang = language.lower()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                if "c++" in lang or "cpp" in lang:
                    src_file = os.path.join(temp_dir, "solution.cpp")
                    exe_file = os.path.join(temp_dir, "solution.exe")

                    with open(src_file, "w", encoding="utf-8") as f:
                        f.write(code)

                    subprocess.check_call(
                        ["g++", "-O2", src_file, "-o", exe_file],
                        stderr=subprocess.DEVNULL,
                    )
                    exec_cmd = [exe_file]

                elif "python" in lang:
                    src_file = os.path.join(temp_dir, "solution.py")

                    with open(src_file, "w", encoding="utf-8") as f:
                        f.write(code)

                    exec_cmd = ["python2" if "2" in lang else "python3", src_file]

                elif "java" in lang:
                    src_file = os.path.join(temp_dir, "Main.java")

                    code = re.sub(
                        r"public\s+class\s+\w+",
                        "public class Main",
                        code,
                    )

                    with open(src_file, "w", encoding="utf-8") as f:
                        f.write(code)

                    subprocess.check_call(
                        ["javac", src_file],
                        stderr=subprocess.DEVNULL,
                    )
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

                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=timeout,
                )

                if process.returncode != 0:
                    return "RUNTIME_ERROR", stderr

                return "SUCCESS", stdout.strip()

            except subprocess.CalledProcessError:
                return "COMPILE_ERROR", ""

            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                return "TIMEOUT", ""

            except Exception as e:
                return "SYSTEM_ERROR", str(e)

    def query_prm(self, description: str, buggy_code: str, sections: dict) -> float:
        payload = {
            "description": description,
            "buggy_code": buggy_code,
            "sections": sections,
            "placeholder_token": self.cfg["prm_placeholder"],
        }

        try:
            resp = requests.post(
                self.cfg["prm_server_url"],
                json=payload,
                timeout=self.cfg["prm_timeout"],
            )
            resp.raise_for_status()
            return float(resp.json()["score"])

        except Exception as e:
            print(f"[WARN] PRM request failed: {repr(e)}")
            return -1.0

    def process_task(self, task_info):
        description = task_info["description"]
        buggy_code = task_info["buggy_code"]
        buggy_lang = task_info["buggy_lang"]
        oracle_code = task_info["oracle_code"]
        oracle_lang = task_info["oracle_lang"]

        task_result = {
            "task_id": task_info.get("task_id", task_info.get("id", "unknown")),
            "pass_extr": 0,
            "pass_pc": 0,
            "pass_bt": 0,
            "pass_pc_bt": 0,
        }

        base_prompt = Prompt.format(description=description, code=buggy_code)

        active_beams = [
            {
                "content": "",
                "sections": {
                    "Bug Analysis": "",
                    "Test Design": "",
                    "Output Prediction": "",
                },
                "score": 0.0,
            }
        ]

        steps = [
            ("Bug Analysis", "### Test Design"),
            ("Test Design", "### Output Prediction"),
            ("Output Prediction", "### Input"),
        ]

        for step_idx, (section_name, stop_token) in enumerate(steps):
            new_candidates = []

            for beam in active_beams:
                assistant_prefix = beam["content"] + f"### {section_name}\n"

                try:
                    response = self.client.chat.completions.create(
                        model=self.cfg["model_name"],
                        messages=[
                            {"role": "user", "content": base_prompt},
                            {"role": "assistant", "content": assistant_prefix},
                        ],
                        temperature=self.cfg["temperature"],
                        max_tokens=self.cfg["step_max_tokens"],
                        n=self.cfg["b2"],
                        stop=[stop_token, "\n" + stop_token],
                        extra_body={
                            "continue_final_message": True,
                            "add_generation_prompt": False,
                            "prefix": True,
                        },
                    )

                except Exception as e:
                    print(
                        f"[WARN] Generation failed at task={task_result['task_id']}, "
                        f"step={section_name}: {repr(e)}"
                    )
                    continue

                for choice in response.choices:
                    gen_text = choice.message.content.strip()
                    if not gen_text:
                        continue

                    new_sections = beam["sections"].copy()
                    new_sections[section_name] = gen_text

                    new_content = assistant_prefix + gen_text + "\n\n"

                    prm_score = self.query_prm(
                        description=description,
                        buggy_code=buggy_code,
                        sections=new_sections,
                    )

                    new_candidates.append(
                        {
                            "content": new_content,
                            "sections": new_sections,
                            "score": prm_score,
                        }
                    )

            if not new_candidates:
                break

            new_candidates.sort(key=lambda x: x["score"], reverse=True)
            active_beams = new_candidates[: self.cfg["b1"]]

        if not active_beams:
            return task_result

        active_beams.sort(key=lambda x: x.get("score", -1.0), reverse=True)
        top_beam = active_beams[0]

        assistant_prefix = top_beam["content"] + "### Input\n"

        try:
            response = self.client.chat.completions.create(
                model=self.cfg["model_name"],
                messages=[
                    {"role": "user", "content": base_prompt},
                    {"role": "assistant", "content": assistant_prefix},
                ],
                temperature=self.cfg["final_temperature"],
                max_tokens=self.cfg["final_max_tokens"],
                n=1,
                extra_body={
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                    "prefix": True,
                },
            )

            final_text = assistant_prefix + response.choices[0].message.content

        except Exception as e:
            print(f"[WARN] Final generation failed at task={task_result['task_id']}: {repr(e)}")
            return task_result

        inp, expected_out = self.extract_xml(final_text)
        if inp is None or expected_out is None:
            return task_result

        task_result["pass_extr"] = 1

        status, oracle_real_out = self.run_code(
            oracle_lang,
            oracle_code,
            inp,
        )

        if status != "SUCCESS":
            return task_result

        is_pred_correct = False
        if self.normalize(oracle_real_out) == self.normalize(expected_out):
            task_result["pass_pc"] = 1
            is_pred_correct = True

        b_status, b_out = self.run_code(
            buggy_lang,
            buggy_code,
            inp,
        )

        triggered = False

        if b_status in ["TIMEOUT", "RUNTIME_ERROR"]:
            triggered = True

        elif b_status == "SUCCESS":
            if self.normalize(b_out) != self.normalize(oracle_real_out):
                triggered = True

        if triggered:
            task_result["pass_bt"] = 1

            if is_pred_correct:
                task_result["pass_pc_bt"] = 1

        return task_result



def init_worker(cfg):
    global GLOBAL_EVALUATOR
    GLOBAL_EVALUATOR = TaskEvaluator(cfg)


def worker_entry(task):
    global GLOBAL_EVALUATOR
    return GLOBAL_EVALUATOR.process_task(task)



def is_task_level_item(item):
    required_fields = ["description", "oracle_code", "oracle_lang", "buggy_code", "buggy_lang"]
    return all(item.get(field) for field in required_fields)


def load_tasks(data_path, max_tasks=-1):
    tasks = []

    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
            except Exception:
                continue

            if is_task_level_item(item):
                task = dict(item)
                task.setdefault("id", item.get("task_id", f"task_{idx}"))
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
                    tasks.append(
                        {
                            "id": f"task_{idx}_bug_{i}",
                            "description": desc,
                            "oracle_code": oracle_code,
                            "oracle_lang": oracle_lang,
                            "buggy_code": b_code,
                            "buggy_lang": b_langs[i] if i < len(b_langs) else "python",
                        }
                    )
                    if max_tasks > 0 and len(tasks) >= max_tasks:
                        return tasks

            if max_tasks > 0 and len(tasks) >= max_tasks:
                return tasks[:max_tasks]

    return tasks


def main():
    args = build_args()
    cfg = vars(args)

    print(f"Loading data from {args.data_path}")
    tasks = load_tasks(args.data_path, max_tasks=args.max_tasks)
    print(f"Loaded {len(tasks)} tasks.")
    print(f"Starting PRM-guided beam search evaluation (B1={args.b1}, B2={args.b2})")

    total_tasks = len(tasks)
    pass_extr_count = 0
    pass_pc_count = 0
    pass_bt_count = 0
    pass_pc_bt_count = 0
    completed = 0
    start_time = time.time()

    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(cfg,),
    ) as pool:
        pbar = tqdm(
            total=total_tasks,
            desc=f"Evaluating B1={args.b1}, B2={args.b2}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )

        for res in pool.imap_unordered(worker_entry, tasks):
            completed += 1
            pass_extr_count += res["pass_extr"]
            pass_pc_count += res["pass_pc"]
            pass_bt_count += res["pass_bt"]
            pass_pc_bt_count += res["pass_pc_bt"]
            pbar.update(1)

            if completed % 50 == 0:
                tqdm.write(
                    f"Progress: {completed}/{total_tasks} Tasks | "
                    f"Extr: {pass_extr_count} | IO Acc: {pass_pc_count} | "
                    f"Attack Rate: {pass_pc_bt_count}"
                )

        pbar.close()

    elapsed = time.time() - start_time
    extr_rate = (pass_extr_count / total_tasks * 100) if total_tasks > 0 else 0.0
    pcr = (pass_pc_count / total_tasks * 100) if total_tasks > 0 else 0.0
    attack_rate = (pass_pc_bt_count / total_tasks * 100) if total_tasks > 0 else 0.0

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"B1:                       {args.b1}")
    print(f"B2:                       {args.b2}")
    print(f"Total Tasks Evaluated:    {total_tasks}")
    print(f"Extraction:               {pass_extr_count} ({extr_rate:.2f}%)")
    print(f"IO Acc:                   {pass_pc_count} ({pcr:.2f}%)")
    print(f"Attack Rate:              {pass_pc_bt_count} ({attack_rate:.2f}%)")
    print(f"Elapsed Time:             {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
