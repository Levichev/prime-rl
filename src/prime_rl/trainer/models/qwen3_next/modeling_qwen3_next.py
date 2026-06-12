import torch
import torch.nn as nn

from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeModel,
)
from prime_rl.trainer.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig


class Qwen3NextGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    """GatedDeltaNet with HF Qwen3-Next fused input projections.

    The math is identical to Qwen3.5's GatedDeltaNet; only the input projections differ.
    Qwen3-Next checkpoints pack them into two fused linears whose rows are interleaved
    per key-head group — ``in_proj_qkvz`` packs ``[q | k | v | z]`` and ``in_proj_ba``
    packs ``[b | a]`` per group. Keeping the fused layout means checkpoint weights load
    1:1 with no conversion.
    """

    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        del self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a
        self.in_proj_qkvz = nn.Linear(self.hidden_size, 2 * self.key_dim + 2 * self.value_dim, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, 2 * self.num_v_heads, bias=False)

    def _project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        heads_per_group = self.num_v_heads // self.num_k_heads
        vz_per_group = heads_per_group * self.head_v_dim

        qkvz = self.in_proj_qkvz(hidden_states).view(
            batch_size, seq_len, self.num_k_heads, 2 * self.head_k_dim + 2 * vz_per_group
        )
        query, key, value, z = torch.split(qkvz, [self.head_k_dim, self.head_k_dim, vz_per_group, vz_per_group], dim=-1)
        ba = self.in_proj_ba(hidden_states).view(batch_size, seq_len, self.num_k_heads, 2 * heads_per_group)
        b, a = torch.split(ba, [heads_per_group, heads_per_group], dim=-1)

        # De-interleave to the flat [q | k | v] x [b, conv_dim, s] layout the conv expects
        mixed_qkv = torch.cat(
            [
                query.reshape(batch_size, seq_len, -1),
                key.reshape(batch_size, seq_len, -1),
                value.reshape(batch_size, seq_len, -1),
            ],
            dim=-1,
        ).transpose(1, 2)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = b.reshape(batch_size, seq_len, self.num_v_heads)
        a = a.reshape(batch_size, seq_len, self.num_v_heads)
        return mixed_qkv, z, b, a


class Qwen3NextDecoderLayer(Qwen3_5MoeDecoderLayer):
    gated_delta_net_cls = Qwen3NextGatedDeltaNet


class Qwen3NextModel(Qwen3_5MoeModel):
    config_class = Qwen3NextConfig
    decoder_layer_cls = Qwen3NextDecoderLayer


class Qwen3NextForCausalLM(Qwen3_5MoeForCausalLM):
    config_class = Qwen3NextConfig
    text_model_cls = Qwen3NextModel

    # Qwen3-Next checkpoints ship multi-token-prediction weights under a top-level
    # `mtp.*` prefix; there is no MTP module here, so drop them on load (mirrors HF).
    _keys_to_ignore_on_load_unexpected = [r"^mtp."]


__all__ = [
    "Qwen3NextForCausalLM",
    "Qwen3NextModel",
    "Qwen3NextGatedDeltaNet",
    "Qwen3NextDecoderLayer",
]
