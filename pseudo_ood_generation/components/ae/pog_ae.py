"""CUDA word-level autoencoder and differentiable sentence decoding."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..sentence_utils import (
    cuda_device, expected_embeddings, positive_temperature, require_device, word_table,
)


def _tokens(tokens, table):
    require_device(tokens, table.weight.device)
    if tokens.dtype != torch.long or tokens.ndim != 2 or min(tokens.shape) < 1:
        raise ValueError("Tokens must be a nonempty [batch, length] CUDA long tensor")
    if ((tokens < 0) | (tokens >= table.num_embeddings)).any():
        raise ValueError("Token ID is outside the vocabulary")


def masked_token_cross_entropy(logits, targets, pad_id=0):
    """Return mean next-token CE over non-PAD targets, or differentiable zero."""
    require_device(targets, logits.device)
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("Expected logits [B,T,V] and long targets [B,T]")
    losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                             ignore_index=pad_id, reduction="none").view_as(targets)
    mask = targets.ne(pad_id)
    return (losses * mask).sum() / mask.sum().clamp_min(1)


class POGEncoder(nn.Module):
    """Read right-padded tokens and select the last valid LSTM output on CUDA."""

    def __init__(self, embedding, code_dim=100):
        super().__init__()
        require_device(embedding.weight, embedding.weight.device)
        if code_dim <= 0:
            raise ValueError("code_dim must be positive")
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.embedding_dim, code_dim, batch_first=True,
                            device=embedding.weight.device)

    def forward(self, tokens):
        _tokens(tokens, self.embedding)
        valid = tokens.ne(self.embedding.padding_idx)
        # Reject internal/left padding; no CPU lengths or packed sequences.
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Encoder tokens must be right padded")
        outputs, _ = self.lstm(self.embedding(tokens))
        lengths = valid.sum(-1)
        indices = (lengths - 1).clamp_min(0)
        codes = outputs.gather(1, indices[:, None, None].expand(-1, 1, outputs.shape[-1]))[:, 0]
        return codes * lengths.gt(0).unsqueeze(-1)


class POGDecoder(nn.Module):
    """Initialize hidden and cell states from the code, then decode word inputs."""

    def __init__(self, embedding, code_dim=100, hidden_dim=None, bos_id=2, eos_id=3):
        super().__init__()
        require_device(embedding.weight, embedding.weight.device)
        hidden_dim = code_dim if hidden_dim is None else hidden_dim
        if min(code_dim, hidden_dim) <= 0:
            raise ValueError("Decoder dimensions must be positive")
        if len({embedding.padding_idx, bos_id, eos_id}) != 3 or not all(
                0 <= token < embedding.num_embeddings for token in (bos_id, eos_id)):
            raise ValueError("PAD, BOS, and EOS must be distinct vocabulary IDs")
        self.embedding = embedding
        self.code_dim, self.hidden_dim = code_dim, hidden_dim
        self.bos_id, self.eos_id = bos_id, eos_id
        device = embedding.weight.device
        self.code_to_state = nn.Linear(code_dim, 2 * hidden_dim, device=device)
        self.lstm = nn.LSTM(embedding.embedding_dim, hidden_dim, batch_first=True, device=device)
        self.projection = nn.Linear(hidden_dim, embedding.num_embeddings, device=device)

    def initial_state(self, codes):
        require_device(codes, self.embedding.weight.device)
        if codes.ndim != 2 or codes.shape[0] < 1 or codes.shape[1] != self.code_dim:
            raise ValueError("Codes must have shape [nonempty batch, code_dim]")
        hidden, cell = self.code_to_state(codes).tanh().chunk(2, dim=-1)
        return hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous()

    def forward(self, codes, input_tokens):
        """Return teacher-forced logits [B,T,V] from decoder input tokens."""
        _tokens(input_tokens, self.embedding)
        if input_tokens.shape[0] != codes.shape[0]:
            raise ValueError("Code and token batch sizes differ")
        outputs, _ = self.lstm(self.embedding(input_tokens), self.initial_state(codes))
        return self.projection(outputs)

    def rollout(self, codes, max_steps=63, temperature=1.0, mode="soft", generator=None):
        """Decode soft feedback or sampled/greedy tokens; keep policy gradients."""
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if mode not in {"soft", "sample", "greedy"}:
            raise ValueError("mode must be soft, sample, or greedy")
        temperature = positive_temperature(temperature)
        state = self.initial_state(codes)
        batch, device = codes.shape[0], codes.device
        inputs = self.embedding(torch.full((batch, 1), self.bos_id, device=device, dtype=torch.long))
        active = torch.ones(batch, device=device, dtype=torch.bool)
        all_logits, all_probs, all_tokens, all_masks, all_log_probs = [], [], [], [], []
        for _ in range(max_steps):
            outputs, state = self.lstm(inputs, state)
            logits = self.projection(outputs[:, 0])
            # All modes prohibit PAD/BOS as outputs; EOS remains eligible.
            allowed = torch.ones(self.embedding.num_embeddings, device=device, dtype=torch.bool)
            allowed[self.embedding.padding_idx] = False
            allowed[self.bos_id] = False
            scaled = (logits / temperature).masked_fill(~allowed, -torch.inf)
            log_probs = scaled.log_softmax(-1)
            probs = scaled.softmax(-1)
            all_logits.append(logits)
            all_probs.append(probs)
            if mode == "soft":
                # No hard EOS stop or argmax in the differentiable path.
                inputs = expected_embeddings(probs, self.embedding).unsqueeze(1)
                all_masks.append(active)
            else:
                selected = (torch.multinomial(probs, 1, generator=generator)[:, 0]
                            if mode == "sample" else probs.argmax(-1))
                selected = torch.where(active, selected, self.embedding.padding_idx)
                all_masks.append(active)
                all_tokens.append(selected)
                # Gather a supported token for inactive rows to avoid 0 * -inf.
                safe_selected = torch.where(active, selected, self.eos_id)
                all_log_probs.append(log_probs.gather(1, safe_selected[:, None])[:, 0] * active)
                active = active & selected.ne(self.eos_id)
                inputs = self.embedding(selected).unsqueeze(1)
        result = {"logits": torch.stack(all_logits, 1),
                  "probabilities": torch.stack(all_probs, 1),
                  "mask": torch.stack(all_masks, 1)}
        if mode != "soft":
            result["tokens"] = torch.stack(all_tokens, 1)
            result["log_probs"] = torch.stack(all_log_probs, 1)
            result["sequence_log_probs"] = result["log_probs"].sum(-1)
        return result


class POGAutoEncoder(nn.Module):
    """One shared word table, a sentence encoder, and an autoregressive decoder."""

    def __init__(self, vocab_size, embedding_dim=100, code_dim=100,
                 decoder_hidden_dim=None, pad_id=0, bos_id=2, eos_id=3,
                 noise_std=1.0, embeddings=None, device="cuda:0"):
        super().__init__()
        device = cuda_device(device)
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        self.pad_id, self.noise_std = pad_id, float(noise_std)
        self.embedding = word_table(vocab_size, embedding_dim, pad_id, device, embeddings)
        self.encoder = POGEncoder(self.embedding, code_dim)
        self.decoder = POGDecoder(self.embedding, code_dim, decoder_hidden_dim, bos_id, eos_id)

    def encode(self, tokens):
        """Return clean [B,code_dim] codes, including zero for all-PAD rows."""
        return self.encoder(tokens)

    def reconstruction_codes(self, codes, add_noise=True, generator=None):
        require_device(codes, self.embedding.weight.device)
        if not add_noise or self.noise_std == 0:
            return codes
        return codes + self.noise_std * torch.randn(
            codes.shape, device=codes.device, dtype=codes.dtype, generator=generator)

    def forward(self, tokens, add_noise=True, temperature=1.0, generator=None):
        """Reconstruct [B,T] tokens (including BOS/EOS) with teacher forcing."""
        _tokens(tokens, self.embedding)
        if tokens.shape[1] < 2:
            raise ValueError("Reconstruction needs at least two token positions")
        temperature = positive_temperature(temperature)
        codes = self.encode(tokens)
        noisy_codes = self.reconstruction_codes(codes, add_noise, generator)
        logits = self.decoder(noisy_codes, tokens[:, :-1])
        targets = tokens[:, 1:]
        return {"codes": codes, "noisy_codes": noisy_codes, "logits": logits,
                "probabilities": (logits / temperature).softmax(-1),
                "mask": targets.ne(self.pad_id),
                "loss": masked_token_cross_entropy(logits, targets, self.pad_id)}

    def rollout(self, codes, max_steps=63, temperature=1.0, mode="soft", generator=None):
        return self.decoder.rollout(codes, max_steps, temperature, mode, generator)
