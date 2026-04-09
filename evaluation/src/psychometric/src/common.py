import os
import math
from typing import Dict, List

import numpy as np
import torch
from torch.nn.functional import log_softmax
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_model_and_tokenizer(model_path: str, dtype: str, device: str):
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            value = _safe_cosine(vectors[i], vectors[j])
            mat[i, j] = value
            mat[j, i] = value
    return mat


@torch.no_grad()
def sequence_logprob(
    model,
    tokenizer,
    text: str,
    device: str,
) -> float:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


@torch.no_grad()
def token_representation_by_layer(
    model,
    tokenizer,
    text: str,
    device: str,
) -> np.ndarray:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states[1:]
    valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if tokenizer.pad_token_id is not None:
        valid_mask &= input_ids != tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        valid_mask &= input_ids != tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        valid_mask &= input_ids != tokenizer.eos_token_id

    token_indices = torch.where(valid_mask[0])[0]
    if token_indices.numel() == 0:
        token_indices = torch.arange(input_ids.shape[1], device=input_ids.device)

    per_layer = []
    for layer_hidden in hidden_states:
        pooled = layer_hidden[0, token_indices, :].mean(dim=0)
        per_layer.append(pooled.detach().to(torch.float32).cpu().numpy())

    return np.stack(per_layer, axis=0)


@torch.no_grad()
def continuation_avg_prob_legacy(
    model,
    tokenizer,
    prompt: str,
    context: str,
    device: str,
    lowercase_prompt: bool = False,
) -> float:
    prompt_text = prompt.lower() if lowercase_prompt else prompt
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    logits = model(input_ids).logits
    all_tokens_logprobs = log_softmax(logits.double(), dim=2)

    token_logprobs = []
    for k in range(0, input_ids.shape[1]):
        token_logprobs.append(float(all_tokens_logprobs[:, k, input_ids[0, k]].item()))

    context_len = len(tokenizer(context, return_tensors="pt").input_ids[0])
    continuation = token_logprobs[context_len:]
    if len(continuation) == 0:
        return float("nan")
    return float(math.exp(sum(continuation) / len(continuation)))


@torch.no_grad()
def continuation_token_logprobs_raven(
    model,
    tokenizer,
    prompt: str,
    context: str,
    device: str,
) -> List[float]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] < 2:
        return []

    logits = model(input_ids).logits
    all_tokens_logprobs = log_softmax(logits.double(), dim=2)

    token_logprobs = []
    for k in range(1, input_ids.shape[1]):
        token_logprobs.append(float(all_tokens_logprobs[:, k - 1, input_ids[0, k]].item()))

    context_len = len(tokenizer(context.strip(), return_tensors="pt").input_ids[0])
    return token_logprobs[context_len:]
