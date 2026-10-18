"""CUDA-only word tables and differentiable token representations."""

import math

import torch
from torch import nn


def cuda_device(device="cuda:0"):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Sentence models require CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("Sentence models require an available CUDA device")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


def require_device(tensor, device):
    if tensor.device != device or tensor.device.type != "cuda":
        raise ValueError(f"Expected a CUDA tensor on {device}, got {tensor.device}")


def word_table(vocab_size, embedding_dim, pad_id, device, embeddings=None):
    if min(vocab_size, embedding_dim) <= 0 or not 0 <= pad_id < vocab_size:
        raise ValueError("Invalid vocabulary, embedding dimension, or PAD ID")
    table = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id, device=device)
    if embeddings is not None:
        require_device(embeddings, device)
        if embeddings.shape != table.weight.shape or not embeddings.is_floating_point():
            raise ValueError("Pretrained embeddings must match the word table shape")
        with torch.no_grad():
            table.weight.copy_(embeddings)
    with torch.no_grad():
        table.weight[pad_id].zero_()
    return table


def positive_temperature(temperature):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return temperature


def annealed_temperature(step, total_steps, start=1.0, floor=0.1):
    start, floor = positive_temperature(start), positive_temperature(floor)
    if floor > start or total_steps <= 0 or step < 0:
        raise ValueError("Invalid temperature schedule")
    return max(floor, start + (floor - start) * min(step / total_steps, 1.0))


def expected_embeddings(probabilities, table, token_mapping=None):
    """Compute q @ E; mapping indexes classifier rows for generator tokens."""
    require_device(probabilities, table.weight.device)
    # padding_idx protects lookups, not direct matrix multiplication.
    rows = torch.arange(table.num_embeddings, device=table.weight.device)
    weight = table.weight * (rows != table.padding_idx).unsqueeze(-1)
    if token_mapping is not None:
        require_device(token_mapping, weight.device)
        if token_mapping.dtype != torch.long or token_mapping.shape != (probabilities.shape[-1],):
            raise ValueError("Token mapping must have one long ID per generator token")
        if ((token_mapping < 0) | (token_mapping >= table.num_embeddings)).any():
            raise ValueError("Token mapping contains out-of-range classifier IDs")
        weight = weight[token_mapping]
    if probabilities.shape[-1] != weight.shape[0]:
        raise ValueError("Token probabilities do not match the embedding vocabulary")
    return probabilities @ weight


def soft_embeddings(logits, table, temperature=1.0, token_mapping=None):
    return expected_embeddings(
        (logits / positive_temperature(temperature)).softmax(-1), table, token_mapping
    )
