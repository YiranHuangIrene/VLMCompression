"""Unfuse / refuse Phi-3 projections for LLM-Pruner compatibility.

Phi-3 ships fused projections: a single ``qkv_proj`` (rows = [Q | K | V])
and a single ``gate_up_proj`` (rows = [gate | up]). LLM-Pruner's dependency
tracer reasons about one ``nn.Linear`` at a time and cannot express the
stripe-pattern pruning the fused layout would require for head removal.

To sidestep that, we temporarily rewrite each ``Phi3Attention`` /
``Phi3MLP`` instance to use separate ``q_proj / k_proj / v_proj`` and
``gate_proj / up_proj`` linears (the LLaMA layout). Width pruning then
reuses the validated ``hf_llama_pruner`` machinery unchanged. Once
pruning is done, ``refuse_phi3`` re-packs the (now smaller) unfused
weights back into the original fused ``qkv_proj`` / ``gate_up_proj``
layout so the saved checkpoint matches the upstream Phi-3 interface.

Only MHA (``num_key_value_heads == num_attention_heads``) is currently
supported — the stripe layout and weight splits assume equal-size Q/K/V
blocks. The helpers raise ``NotImplementedError`` on GQA/MQA configs.
"""

import types

import torch
import torch.nn as nn


def _assert_mha(attn):
    if attn.num_key_value_heads != attn.num_heads:
        raise NotImplementedError(
            f"Phi-3 unfuse/refuse currently supports MHA only; got "
            f"num_heads={attn.num_heads} vs num_key_value_heads={attn.num_key_value_heads}"
        )


def _unfused_attention_forward(self, hidden_states, attention_mask=None,
                               position_ids=None, past_key_value=None,
                               output_attentions=False, use_cache=False):
    """Drop-in Phi3Attention.forward for the unfused layout.

    Mirrors the original forward in modeling_phi3.py line-for-line, with
    the three fused slices replaced by direct calls to q/k/v_proj.
    """
    from LLMPruner.models.hf_phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv
    import math

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def _unfused_mlp_forward(self, hidden_states):
    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    return self.down_proj(up * self.activation_fn(gate))


def unfuse_phi3(model):
    """Rewrite every ``Phi3Attention`` / ``Phi3MLP`` in ``model`` to use
    unfused projections. Mutates ``model`` in place and returns it."""
    for layer in model.model.layers:
        attn = layer.self_attn
        _assert_mha(attn)

        qkv_w = attn.qkv_proj.weight.data
        H, D = attn.num_heads, attn.head_dim
        hidden = attn.hidden_size
        q_size = H * D
        kv_size = attn.num_key_value_heads * D
        assert qkv_w.shape[0] == q_size + 2 * kv_size

        dev, dtype = qkv_w.device, qkv_w.dtype
        q_proj = nn.Linear(hidden, q_size, bias=False, device=dev, dtype=dtype)
        k_proj = nn.Linear(hidden, kv_size, bias=False, device=dev, dtype=dtype)
        v_proj = nn.Linear(hidden, kv_size, bias=False, device=dev, dtype=dtype)
        q_proj.weight.data.copy_(qkv_w[:q_size])
        k_proj.weight.data.copy_(qkv_w[q_size : q_size + kv_size])
        v_proj.weight.data.copy_(qkv_w[q_size + kv_size :])

        attn.q_proj = q_proj
        attn.k_proj = k_proj
        attn.v_proj = v_proj
        del attn.qkv_proj
        attn.forward = types.MethodType(_unfused_attention_forward, attn)

        mlp = layer.mlp
        gu_w = mlp.gate_up_proj.weight.data
        I = mlp.config.intermediate_size
        hidden_mlp = mlp.config.hidden_size
        assert gu_w.shape[0] == 2 * I
        gate_proj = nn.Linear(hidden_mlp, I, bias=False, device=gu_w.device, dtype=gu_w.dtype)
        up_proj = nn.Linear(hidden_mlp, I, bias=False, device=gu_w.device, dtype=gu_w.dtype)
        gate_proj.weight.data.copy_(gu_w[:I])
        up_proj.weight.data.copy_(gu_w[I:])
        mlp.gate_proj = gate_proj
        mlp.up_proj = up_proj
        del mlp.gate_up_proj
        mlp.forward = types.MethodType(_unfused_mlp_forward, mlp)

    return model


def refuse_phi3(model):
    """Re-pack unfused projections back into the fused ``qkv_proj`` /
    ``gate_up_proj`` layout and restore the original forward methods.
    Mutates ``model`` in place and returns it."""
    from LLMPruner.models.hf_phi3.modeling_phi3 import Phi3Attention, Phi3MLP

    for layer in model.model.layers:
        attn = layer.self_attn
        q_w = attn.q_proj.weight.data
        k_w = attn.k_proj.weight.data
        v_w = attn.v_proj.weight.data

        new_q_size = q_w.shape[0]
        new_kv_size = k_w.shape[0]
        new_hidden = q_w.shape[1]
        assert k_w.shape[0] == v_w.shape[0], "K and V must share size after pruning"

        # Per-layer shapes may have changed — update the attention attrs.
        attn.head_dim = new_q_size // attn.num_heads  # invariant under head pruning, recomputed defensively
        new_num_heads = new_q_size // attn.head_dim
        attn.num_heads = new_num_heads
        attn.num_key_value_heads = new_kv_size // attn.head_dim
        attn.num_key_value_groups = attn.num_heads // attn.num_key_value_heads
        attn.hidden_size = new_hidden

        op_size = new_q_size + 2 * new_kv_size
        qkv_proj = nn.Linear(new_hidden, op_size, bias=False,
                             device=q_w.device, dtype=q_w.dtype)
        qkv_proj.weight.data[:new_q_size].copy_(q_w)
        qkv_proj.weight.data[new_q_size : new_q_size + new_kv_size].copy_(k_w)
        qkv_proj.weight.data[new_q_size + new_kv_size :].copy_(v_w)
        attn.qkv_proj = qkv_proj
        del attn.q_proj, attn.k_proj, attn.v_proj
        # Restore the class-level forward (undo the monkey-patched method).
        if 'forward' in attn.__dict__:
            del attn.__dict__['forward']

        mlp = layer.mlp
        g_w = mlp.gate_proj.weight.data
        u_w = mlp.up_proj.weight.data
        assert g_w.shape == u_w.shape, "gate and up must share shape after pruning"
        new_I = g_w.shape[0]
        new_hidden_mlp = g_w.shape[1]
        gate_up_proj = nn.Linear(new_hidden_mlp, 2 * new_I, bias=False,
                                 device=g_w.device, dtype=g_w.dtype)
        gate_up_proj.weight.data[:new_I].copy_(g_w)
        gate_up_proj.weight.data[new_I:].copy_(u_w)
        mlp.gate_up_proj = gate_up_proj
        del mlp.gate_proj, mlp.up_proj
        if 'forward' in mlp.__dict__:
            del mlp.__dict__['forward']

    return model
