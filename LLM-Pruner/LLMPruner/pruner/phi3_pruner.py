"""Phi-3 pruner.

After :func:`LLMPruner.models.hf_phi3.fusion.unfuse_phi3` rewrites the
fused ``qkv_proj`` / ``gate_up_proj`` into separate ``q_proj / k_proj /
v_proj`` and ``gate_proj / up_proj`` linears, Phi-3's attention and MLP
layout is identical to LLaMA's. We therefore re-export the LLaMA
pruner's importance classes and custom pruners verbatim, and only add
a ``Phi3RMSNorm``-keyed alias so ``customized_pruners`` can register it.
"""

from LLMPruner.pruner.hf_llama_pruner import (  # noqa: F401
    HFRMSNormPrunner,
    HFAttentionPrunner,
    HFLinearPrunner,
    MagnitudeImportance,
    TaylorImportance,
    hf_attention_pruner,
    hf_linear_pruner,
    hf_rmsnorm_pruner,
)
