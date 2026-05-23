#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

import requests
MIN_SECTION_WORDS = 15

def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))

def has_enough_words_per_section(sections: Dict[str, str], min_words: int) -> bool:
    for section_name in ("Bug Analysis", "Test Design", "Output Prediction"):
        if count_words(normalize_text(sections.get(section_name))) < min_words:
            return False
    return True
def extract_block(start_header, end_header, content):
    if end_header:
        pattern = rf"(?:###|##|\*\*|{start_header}:?)\s*{start_header}[:\s]*\n+(.*?)(?=\n\s*(?:###|##|\*\*|{end_header}))"
    else:
        pattern = rf"(?:###|##|\*\*|{start_header}:?)\s*{start_header}[:\s]*\n+(.*)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).replace("```", "").replace("<code>", "").replace("</code>", "").strip()
    return None

def normalize(s):
    if not s:
        return ""
    return "\n".join([line.rstrip() for line in s.strip().splitlines()])

def run_code(language, code, input_data, timeout=3):
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
                with open(src_file, "w", encoding="utf-8") as f:
                    code = re.sub(r"public\s+class\s+\w+", "public class Main", code)
                    f.write(code)
                subprocess.check_call(["javac", src_file], stderr=subprocess.DEVNULL)
                exec_cmd = ["java", "-cp", temp_dir, "Main"]
            else:
                return "UNSUPPORTED", ""

            process = subprocess.Popen(
                exec_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, errors="ignore"
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

REQUIRED_STEPS = ["### Bug Analysis", "### Test Design", "### Output Prediction", "### Input", "### Expected Output"]

def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()

def extract_reasoning_sections(solution_str: str) -> Optional[Dict[str, str]]:
    sections = {
        "Bug Analysis": extract_block("Bug Analysis", "Test Design", solution_str),
        "Test Design": extract_block("Test Design", "Output Prediction", solution_str),
        "Output Prediction": extract_block("Output Prediction", "Input", solution_str),
    }
    if any(not normalize_text(v) for v in sections.values()):
        return None
    return sections

def query_prm_server(server_url: str, description: str, buggy_code: str, sections: Dict[str, str], placeholder: str, timeout: float) -> Tuple[float, Dict]:
    payload = {
        "description": description,
        "buggy_code": buggy_code,
        "sections": sections,
        "placeholder_token": placeholder,
    }
    resp = requests.post(server_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return float(data["score"]), data

def compute_score(solution_str: str, ground_truth: dict, **kwargs):
    desc = ground_truth.get("description", "")
    o_code = ground_truth.get("oracle_code", "")
    o_lang = ground_truth.get("oracle_lang", "")
    b_code = ground_truth.get("buggy_code", "")
    b_lang = ground_truth.get("buggy_lang", "")

    prm_server_url = kwargs.get("prm_server_url") or os.getenv("PRM_SERVER_URL", "")
    prm_timeout = float(kwargs.get("prm_timeout") or os.getenv("PRM_TIMEOUT", "10"))
    prm_placeholder_token = kwargs.get("prm_placeholder_token") or os.getenv("PRM_PLACEHOLDER_TOKEN", "ки")

    if not prm_server_url:
        return {"score": -1.0, "acc": False, "reward_prm": 0.0}

    def fail_result():
        return {
            "score": -1.0,
            "acc": False,
            "reward_prm": 0.0,
        }

    last_idx = -1
    for step in REQUIRED_STEPS:
        idx = solution_str.find(step)
        if idx == -1 or idx <= last_idx:
            return fail_result()
        last_idx = idx

    inp = extract_block("Input", "Expected Output", solution_str)
    expected_out = extract_block("Expected Output", None, solution_str)
    reasoning_sections = extract_reasoning_sections(solution_str)
    if not inp or not expected_out or reasoning_sections is None:
        return fail_result()

    if any(
        not normalize_text(reasoning_sections.get(section_name))
        for section_name in ("Bug Analysis", "Test Design", "Output Prediction")
    ):
        return fail_result()

    min_section_words = int(
        kwargs.get("min_section_words")
        or os.getenv("MIN_SECTION_WORDS", str(MIN_SECTION_WORDS))
    )
    if not has_enough_words_per_section(reasoning_sections, min_section_words):
        return fail_result()

    norm_inp = " ".join(inp.strip().split())
    official_examples = []
    matches = re.findall(
        r"Input\s*\n+(.*?)\n+\s*Output",
        desc,
        re.IGNORECASE | re.DOTALL,
    )
    for m in matches:
        clean_m = " ".join(m.strip().split())
        if clean_m:
            official_examples.append(clean_m)

    is_plagiarized = False
    if official_examples:
        if norm_inp in official_examples:
            is_plagiarized = True
    else:
        norm_desc = " ".join(desc.strip().split())
        if len(norm_inp) > 5 and norm_inp in norm_desc:
            is_plagiarized = True
    if is_plagiarized:
        return fail_result()

    o_status, real_out = run_code(o_lang, o_code, inp, timeout=2)
    if o_status != "SUCCESS":
        return fail_result()

    is_pred_correct = (normalize(expected_out) == normalize(real_out))

    b_status, b_out = run_code(b_lang, b_code, inp, timeout=2)
    is_bug_triggered = False
    if b_status == "TIMEOUT":
        is_bug_triggered = True
    elif b_status == "SUCCESS" and normalize(b_out) != normalize(real_out):
        is_bug_triggered = True

    full_success = is_pred_correct and is_bug_triggered

    # Outcome reward used as the PAPO score.
    if full_success:
        exec_reward = 1.0
    else:
        exec_reward = 0.0
        if is_pred_correct:
            exec_reward += 0.2
        if is_bug_triggered:
            exec_reward += 0.2

    # Correctness mask used as the PAPO accuracy signal.
    acc = bool(full_success)

    # Process reward used as the PAPO PRM signal.
    prm_reward = 0.0
    acc=False
    # if acc:
    #     try:
    #         prm_reward, _ = query_prm_server(
    #             prm_server_url,
    #             desc,
    #             b_code,
    #             reasoning_sections,
    #             prm_placeholder_token,
    #             prm_timeout,
    #         )
    #         prm_reward = max(0.0, min(1.0, float(prm_reward)))
    #     except Exception:
    #         prm_reward = exec_reward

    return {
        "score": float(exec_reward),
        "acc": acc,
        "reward_prm": float(prm_reward),
    }
    
