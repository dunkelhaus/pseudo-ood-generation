"""Independent CUDA word table and parallel, mask-safe intent CNN."""

import torch
from torch import nn
from torch.nn import functional as F

from .sentence_utils import cuda_device, expected_embeddings, require_device, soft_embeddings, word_table


class POGAuxilliaryClassifier(nn.Module):
    """Predict IND intents from discrete tokens or expected word embeddings."""

    def __init__(self, vocab_size, embedding_dim=100, num_classes=150,
                 out_channels=128, kernel_sizes=(2, 3, 4, 5), mlp_hidden_size=512,
                 pad_id=0, embeddings=None, device="cuda:0"):
        super().__init__()
        device = cuda_device(device)
        if min(num_classes, out_channels, mlp_hidden_size) <= 0:
            raise ValueError("Classifier dimensions must be positive")
        if not kernel_sizes or any(not isinstance(k, int) or k <= 0 for k in kernel_sizes):
            raise ValueError("Kernel widths must be positive integers")
        self.pad_id, self.kernel_sizes = pad_id, tuple(kernel_sizes)
        self.embedding = word_table(vocab_size, embedding_dim, pad_id, device, embeddings)
        self.cnn = nn.ModuleList([
            nn.Conv1d(embedding_dim, out_channels, width, device=device)
            for width in self.kernel_sizes
        ])
        self.mlp1 = nn.Linear(out_channels * len(self.kernel_sizes), mlp_hidden_size, device=device)
        self.mlp2 = nn.Linear(mlp_hidden_size, mlp_hidden_size, device=device)
        self.mlp3 = nn.Linear(mlp_hidden_size, num_classes, device=device)

    def forward_embeddings(self, embeddings, mask):
        """Classify [B,T,E]; a valid CNN window must contain no masked tokens."""
        require_device(embeddings, self.embedding.weight.device)
        require_device(mask, embeddings.device)
        if (embeddings.ndim != 3 or min(embeddings.shape[:2]) < 1
                or embeddings.shape[-1] != self.embedding.embedding_dim
                or mask.dtype != torch.bool or mask.shape != embeddings.shape[:2]):
            raise ValueError("Expected embeddings [B,T,E] and boolean mask [B,T]")
        embeddings = embeddings * mask.unsqueeze(-1)
        extra = max(0, max(self.kernel_sizes) - embeddings.shape[1])
        inputs = F.pad(embeddings.transpose(1, 2), (0, extra))
        mask = F.pad(mask, (0, extra), value=False)
        pooled = []
        for width, convolution in zip(self.kernel_sizes, self.cnn):
            features = F.relu(convolution(inputs))
            valid = mask.unfold(1, width, 1).all(-1)
            maximum = features.masked_fill(~valid[:, None, :], -torch.inf).amax(-1)
            pooled.append(torch.where(valid.any(-1, keepdim=True), maximum, 0.0))
        features = torch.cat(pooled, -1)
        return self.mlp3(F.relu(self.mlp2(F.relu(self.mlp1(features)))))

    def forward(self, tokens, mask=None):
        require_device(tokens, self.embedding.weight.device)
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("Expected long token IDs [B,T]")
        if ((tokens < 0) | (tokens >= self.embedding.num_embeddings)).any():
            raise ValueError("Token ID is outside the classifier vocabulary")
        valid = tokens.ne(self.pad_id)
        if mask is not None:
            require_device(mask, tokens.device)
            if mask.shape != tokens.shape or mask.dtype != torch.bool:
                raise ValueError("Expected boolean token mask [B,T]")
            valid = valid & mask
        return self.forward_embeddings(self.embedding(tokens), valid)

    def forward_soft(self, logits, mask, temperature=1.0, token_mapping=None):
        return self.forward_embeddings(
            soft_embeddings(logits, self.embedding, temperature, token_mapping), mask)

    def forward_probabilities(self, probabilities, mask, token_mapping=None):
        return self.forward_embeddings(
            expected_embeddings(probabilities, self.embedding, token_mapping), mask)


# Keep the original public spelling while offering the standard spelling.
POGAuxiliaryClassifier = POGAuxilliaryClassifier
SentenceCNN = POGAuxilliaryClassifier
