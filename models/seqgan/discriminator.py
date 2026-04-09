from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiscriminatorConfig:
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    pad_token_id: int = 0


class SeqGANDiscriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
        )
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(config.hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        states, _ = self.lstm(embedded)
        pooled = states[:, -1, :]
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

    @torch.inference_mode()
    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return torch.sigmoid(logits)
