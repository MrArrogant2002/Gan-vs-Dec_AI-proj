from __future__ import annotations

import copy

import torch

from models.seqgan.discriminator import SeqGANDiscriminator
from models.seqgan.generator import SeqGANGenerator


class RolloutPolicy:
    def __init__(self, generator: SeqGANGenerator, update_rate: float = 0.8) -> None:
        self.generator = copy.deepcopy(generator)
        self.update_rate = update_rate

    def update_params(self, source_generator: SeqGANGenerator) -> None:
        for target_param, source_param in zip(self.generator.parameters(), source_generator.parameters()):
            target_param.data.mul_(self.update_rate).add_(source_param.data * (1.0 - self.update_rate))

    @torch.inference_mode()
    def get_rewards(
        self,
        sampled_ids: torch.Tensor,
        rollout_num: int,
        discriminator: SeqGANDiscriminator,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size, seq_len = sampled_ids.size()
        device = sampled_ids.device
        rewards = torch.zeros(batch_size, seq_len - 1, device=device)

        for rollout_index in range(rollout_num):
            for prefix_length in range(1, seq_len):
                prefixes = sampled_ids[:, :prefix_length]
                completed = self.generator.complete_sequences(
                    prefixes=prefixes,
                    target_len=seq_len,
                    temperature=temperature,
                )
                scores = discriminator.predict_proba(completed)
                rewards[:, prefix_length - 1] += scores

        rewards /= max(rollout_num, 1)
        return rewards
