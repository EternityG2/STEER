#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import queue
from typing import Dict, List, Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

from openrlhf.utils.utils import convert_token_to_id
from prompts import Prompt as PROMPT_TEMPLATE


app = FastAPI()

TOKENIZER = None
MODEL_WORKERS = []
AVAILABLE_WORKERS = None
PLACEHOLDER_DEFAULT = "ки"
PLACEHOLDER_TOKEN_ID = None
REWARD_TOKENS = ["+", "-"]
REWARD_TOKEN_IDS = None
ID_TO_REWARD = None
AGGREGATE_MODE = "mean"
MAX_LEN = 4096


class ScoreRequest(BaseModel):
    description: str
    buggy_code: str
    sections: Dict[str, str]
    placeholder_token: str = PLACEHOLDER_DEFAULT


def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def build_prompt(description: str, buggy_code: str) -> str:
    return PROMPT_TEMPLATE.format(description=normalize_text(description), code=str(buggy_code).rstrip())


def build_candidate_prefix(sections: Dict[str, str], placeholder: str) -> str:
    return (
        "### Bug Analysis\n"
        f"{normalize_text(sections['Bug Analysis'])}\n{placeholder}\n"
        "### Test Design\n"
        f"{normalize_text(sections['Test Design'])}\n{placeholder}\n"
        "### Output Prediction\n"
        f"{normalize_text(sections['Output Prediction'])}\n{placeholder}\n"
    )


def softmax_two(a: float, b: float) -> Tuple[float, float]:
    x = max(a, b)
    ea = torch.exp(torch.tensor(a - x))
    eb = torch.exp(torch.tensor(b - x))
    z = ea + eb
    return float(ea / z), float(eb / z)


def score_from_logits(logits: torch.Tensor, placeholder_mask: torch.Tensor) -> Dict:
    masked_logits = logits[placeholder_mask]
    masked_logits = masked_logits[..., REWARD_TOKEN_IDS]
    pred_class = masked_logits.argmax(dim=-1)

    step_probs = []
    pred_labels = []
    scores = []

    plus_idx = 0
    minus_idx = 1
    for row, cls in zip(masked_logits, pred_class):
        plus_logit = float(row[plus_idx].item())
        minus_logit = float(row[minus_idx].item())
        p_plus, p_minus = softmax_two(plus_logit, minus_logit)
        step_probs.append(p_plus)
        pred_labels.append(ID_TO_REWARD[REWARD_TOKEN_IDS[int(cls.item())]])
        scores.append({"+": plus_logit, "-": minus_logit, "p_plus": p_plus, "p_minus": p_minus})

    if not step_probs:
        raise ValueError("no placeholder token remains after truncation")

    if AGGREGATE_MODE == "min":
        score = min(step_probs)
    elif AGGREGATE_MODE == "last":
        score = step_probs[-1]
    elif AGGREGATE_MODE == "weighted":
        weights = [0.4, 0.35, 0.25]
        use_weights = weights[: len(step_probs)]
        if len(use_weights) < len(step_probs):
            tail = len(step_probs) - len(use_weights)
            use_weights.extend([1.0 / len(step_probs)] * tail)
        norm = sum(use_weights)
        score = sum(p * w for p, w in zip(step_probs, use_weights)) / norm
    else:
        score = sum(step_probs) / len(step_probs)

    return {
        "score": float(score),
        "aggregate": AGGREGATE_MODE,
        "step_probs": step_probs,
        "pred_labels": pred_labels,
        "scores": scores,
    }


def encode_request(req: ScoreRequest):
    prompt = build_prompt(req.description, req.buggy_code)
    candidate = build_candidate_prefix(req.sections, req.placeholder_token)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": candidate}]
    
    input_text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    enc = TOKENIZER(
        input_text,
        max_length=MAX_LEN,
        padding=False,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    placeholder_mask = input_ids == PLACEHOLDER_TOKEN_ID
    if int(placeholder_mask.sum().item()) == 0:
        raise ValueError("no placeholder token found in truncated input")
    return enc, input_ids, placeholder_mask


def acquire_worker():
    worker_idx = AVAILABLE_WORKERS.get()
    return worker_idx, MODEL_WORKERS[worker_idx]


def release_worker(worker_idx: int):
    AVAILABLE_WORKERS.put(worker_idx)


def score_one(req: ScoreRequest) -> Dict:
    if req.placeholder_token != PLACEHOLDER_DEFAULT:
        raise ValueError(
            f"server placeholder_token is fixed to {PLACEHOLDER_DEFAULT!r}, got {req.placeholder_token!r}"
        )

    enc, input_ids, placeholder_mask = encode_request(req)
    worker_idx, worker = acquire_worker()
    try:
        model_inputs = {
            "input_ids": enc["input_ids"].to(worker["device"]),
            "attention_mask": enc["attention_mask"].to(worker["device"]),
        }
        with torch.inference_mode():
            outputs = worker["model"](**model_inputs)
            logits = outputs.logits.float().cpu()[0]
        return score_from_logits(logits, placeholder_mask[0])
    finally:
        release_worker(worker_idx)


@app.post("/score")
def score_endpoint(req: ScoreRequest):
    return score_one(req)


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": ",".join(str(worker["device"]) for worker in MODEL_WORKERS),
        "aggregate": AGGREGATE_MODE,
    }


def load_model_on_device(model_path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model


def main():
    global TOKENIZER, MODEL_WORKERS, AVAILABLE_WORKERS
    global PLACEHOLDER_DEFAULT, PLACEHOLDER_TOKEN_ID, REWARD_TOKENS, REWARD_TOKEN_IDS, ID_TO_REWARD
    global AGGREGATE_MODE, MAX_LEN

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--placeholder_token", type=str, default="ки")
    parser.add_argument("--reward_tokens", type=str, nargs=2, default=["+", "-"])
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "min", "last", "weighted"])
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1",
        help="Comma-separated CUDA device indices used for replicated PRM workers.",
    )
    args = parser.parse_args()

    PLACEHOLDER_DEFAULT = args.placeholder_token
    REWARD_TOKENS = args.reward_tokens
    AGGREGATE_MODE = args.aggregate
    MAX_LEN = args.max_len

    tokenizer_path = args.tokenizer_path or args.model_path
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id

    PLACEHOLDER_TOKEN_ID = convert_token_to_id(PLACEHOLDER_DEFAULT, TOKENIZER)
    REWARD_TOKEN_IDS = [convert_token_to_id(tok, TOKENIZER) for tok in REWARD_TOKENS]
    ID_TO_REWARD = {tid: tok for tid, tok in zip(REWARD_TOKEN_IDS, REWARD_TOKENS)}

    if not torch.cuda.is_available():
        raise RuntimeError("prm_server.py expects CUDA, but torch.cuda.is_available() is False.")

    requested_devices = [x.strip() for x in args.cuda_devices.split(",") if x.strip()]
    if len(requested_devices) != 2:
        raise ValueError(f"expected exactly 2 CUDA devices, got {requested_devices}")

    dtype = torch.bfloat16
    MODEL_WORKERS = []
    AVAILABLE_WORKERS = queue.Queue()

    for worker_idx, dev_idx in enumerate(requested_devices):
        device = f"cuda:{dev_idx}"
        print(f"Loading PRM worker {worker_idx} on {device} ...", flush=True)
        model = load_model_on_device(args.model_path, device, dtype)
        MODEL_WORKERS.append({"device": device, "model": model})
        AVAILABLE_WORKERS.put(worker_idx)

    print(
        f"PRM server ready. devices={[worker['device'] for worker in MODEL_WORKERS]} "
        f"placeholder_token={PLACEHOLDER_DEFAULT!r} reward_tokens={REWARD_TOKENS} max_len={MAX_LEN}",
        flush=True,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
