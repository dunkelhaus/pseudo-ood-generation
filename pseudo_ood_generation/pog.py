"""Sentence-model container; optimization belongs to the phase trainer."""

import torch
from torch import nn


class POG(nn.Module):
    """Keep sentence models and fork-owned latent GAN modules in one checkpoint."""

    def __init__(self, autoencoder, auxiliary, generator, critic, token_mapping):
        super().__init__()
        self.autoencoder, self.auxiliary = autoencoder, auxiliary
        self.generator, self.critic = generator, critic
        device = autoencoder.embedding.weight.device
        if device.type != "cuda":
            raise ValueError("POG requires CUDA")
        if (token_mapping.device != device or token_mapping.dtype != torch.long
                or token_mapping.shape != (autoencoder.embedding.num_embeddings,)):
            raise ValueError("token_mapping must be a CUDA long ID per generator token")
        if ((token_mapping < 0) | (token_mapping >= auxiliary.embedding.num_embeddings)).any():
            raise ValueError("token_mapping contains invalid auxiliary vocabulary IDs")
        if token_mapping[autoencoder.pad_id].item() != auxiliary.pad_id:
            raise ValueError("Generator PAD must map to auxiliary PAD")
        if not (autoencoder.decoder.code_dim == generator.code_dim == critic.code_dim):
            raise ValueError("AE, generator, and critic code dimensions must match")
        seen = set()
        for module in self.phase_modules.values():
            ids = {id(p) for p in module.parameters()}
            if ids & seen:
                raise ValueError("Phase modules must not share parameters")
            seen.update(ids)
            if any(p.device != device for p in module.parameters()):
                raise ValueError("All POG parameters must share one CUDA device")
        self.register_buffer("token_mapping", token_mapping.detach().clone())

    @property
    def phase_modules(self):
        return {"ae": self.autoencoder, "auxiliary": self.auxiliary,
                "generator": self.generator, "critic": self.critic}
