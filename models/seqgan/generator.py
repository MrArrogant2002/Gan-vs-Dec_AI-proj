from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class GeneratorConfig:
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    pad_token_id: int = 0


class SeqGANGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig) -> None:
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
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embeddings = self.embedding(input_ids)
        states, hidden = self.lstm(embeddings, hidden)
        logits = self.output(states)
        return logits, hidden

    def pretrain_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(input_ids)
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.config.pad_token_id,
        )

    def policy_gradient_loss(
        self,
        sampled_ids: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward(sampled_ids[:, :-1])
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=sampled_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        loss = -(chosen_log_probs * rewards).mean()
        return loss

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        bos_token_id: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        hidden = None

        for _ in range(seq_len - 1):
            logits, hidden = self.forward(generated[:, -1:].to(device), hidden)
            step_logits = logits[:, -1, :] / max(temperature, 1e-5)
            distribution = Categorical(logits=step_logits)
            next_tokens = distribution.sample().unsqueeze(-1)
            generated = torch.cat([generated, next_tokens], dim=1)

            if eos_token_id is not None and torch.all(next_tokens.squeeze(-1) == eos_token_id):
                break

        return generated

    @torch.inference_mode()
    def complete_sequences(
        self,
        prefixes: torch.Tensor,
        target_len: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = prefixes.device
        generated = prefixes.clone()
        hidden = None

        if prefixes.size(1) > 1:
            _, hidden = self.forward(prefixes[:, :-1])

        current = prefixes[:, -1:]
        while generated.size(1) < target_len:
            logits, hidden = self.forward(current, hidden)
            step_logits = logits[:, -1, :] / max(temperature, 1e-5)
            distribution = Categorical(logits=step_logits)
            current = distribution.sample().unsqueeze(-1)
            generated = torch.cat([generated, current.to(device)], dim=1)
        return generated
