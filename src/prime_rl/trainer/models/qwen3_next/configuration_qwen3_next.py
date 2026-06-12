from prime_rl.trainer.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig


class Qwen3NextConfig(Qwen3_5MoeConfig):
    r"""
    Configuration class for the custom PrimeRL Qwen3-Next model.

    Qwen3-Next is the architectural predecessor of Qwen3.5-MoE: the hybrid layer schedule
    (GatedDeltaNet linear + gated softmax attention), partial RoPE, MoE with gated shared
    expert, and (1+weight) RMSNorm are identical. The differences are config defaults and
    the HF checkpoint packing of the DeltaNet input projections (fused ``in_proj_qkvz`` /
    ``in_proj_ba`` instead of four flat projections).

    Defaults match Qwen3-Next-80B-A3B.
    """

    model_type = "qwen3_next"

    def __init__(
        self,
        vocab_size=151936,
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        rope_theta=10000000.0,
        # HF Qwen3-Next fields without a Qwen3.5 counterpart
        intermediate_size=5632,
        decoder_sparse_step=1,
        mlp_only_layers=None,
        norm_topk_prob=True,
        **kwargs,
    ):
        # transformers 5.x checkpoints may carry rope settings as a `rope_parameters` dict
        rope_parameters = kwargs.pop("rope_parameters", None)
        if rope_parameters:
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
            if partial_rotary_factor is not None:
                kwargs.setdefault("partial_rotary_factor", partial_rotary_factor)

        self.intermediate_size = intermediate_size
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = list(mlp_only_layers) if mlp_only_layers else []
        self.norm_topk_prob = norm_topk_prob

        if self.decoder_sparse_step != 1 or self.mlp_only_layers:
            raise ValueError(
                "Custom Qwen3Next only supports all-MoE checkpoints (decoder_sparse_step=1 and empty mlp_only_layers)."
            )
        if not self.norm_topk_prob:
            raise ValueError("Custom Qwen3Next requires norm_topk_prob=True (top-k renormalization is hardcoded).")

        super().__init__(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            rope_theta=rope_theta,
            **kwargs,
        )
