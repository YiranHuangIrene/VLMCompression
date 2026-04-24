"""Width pruning for Mini-InternVL-Chat-4B-V1-5's Phi-3 language backbone.

Flow (see LLMPruner/models/hf_phi3/fusion.py for the rationale):
    1. Load the bare Phi3ForCausalLM weights from the Mini-InternVL checkpoint.
    2. :func:`unfuse_phi3` rewrites every ``qkv_proj`` / ``gate_up_proj`` as
       separate ``q_proj / k_proj / v_proj`` and ``gate_proj / up_proj`` linears.
       In that layout, Phi-3 is structurally identical to LLaMA, so LLM-Pruner's
       validated ``hf_llama_pruner`` machinery applies unchanged.
    3. Run MetaPruner with block-wise head + MLP-intermediate pruning.
    4. :func:`refuse_phi3` packs the pruned unfused weights back into the fused
       ``qkv_proj`` / ``gate_up_proj`` layout expected by stock Phi-3.
    5. Save as ``{'model': phi3, 'tokenizer': tokenizer}`` — bare Phi-3 format
       matching Bunny.py / llava-vicuna_prune.py and consumed by the InternVL
       recovery loader at ``VLM/InternVL/internvl_chat/internvl/model/internvl_chat/builder.py``.
"""
import os
import gc
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import argparse
import copy
import random

import numpy as np
import torch
from transformers import AutoTokenizer

import LLMPruner.torch_pruning as tp
from LLMPruner.models.hf_phi3.modeling_phi3 import (
    Phi3ForCausalLM,
    Phi3RMSNorm,
)
from LLMPruner.models.hf_phi3.fusion import unfuse_phi3, refuse_phi3
from LLMPruner.pruner import phi3_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_phi3_from_internvl(base_model, torch_dtype):
    """Load the Phi-3 language backbone out of a Mini-InternVL checkpoint.

    Mini-InternVL ships the full InternVLChatModel; we only need its
    ``.language_model`` submodule. Loading the whole VLM just to throw
    away the ViT + projector is wasteful, so we let HF's ``from_pretrained``
    load Phi-3 directly — it will pick up the embedded language_model
    weights via ``base_model_prefix='language_model'``.
    """
    return Phi3ForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    )


def main(args):
    set_random_seed(args.seed)

    log_name = "{}_{}_{}".format(
        args.base_model.rstrip('/').split('/')[-1],
        args.pruning_ratio,
        args.dataset,
    )
    if args.iterative_steps > 1:
        log_name += "_iter_{}".format(args.iterative_steps)
    if args.num_examples != 10:
        log_name += "_{}_samples".format(args.num_examples)
    logger = LoggerWithDepth(
        env_name=log_name,
        config=args.__dict__,
        root_dir=os.path.join(os.path.dirname(__file__), '../LLMPruner/prune_log'),
        setup_sublogger=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, add_eos_token=False, trust_remote_code=True, use_fast=True)

    logger.log("Loading Phi-3 backbone from {} ...".format(args.base_model))
    model = _load_phi3_from_internvl(args.base_model, torch_dtype=torch.float32)
    logger.log("Unfusing qkv_proj / gate_up_proj -> q/k/v_proj and gate/up_proj")
    unfuse_phi3(model)
    model.to(args.device)

    # Set layer_idx defensively (some HF loaders skip it).
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("#Param before: {}".format(before_pruning_parameters))

    # Any input builds the dep graph; actual values are irrelevant.
    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ]).to(args.device)

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']
    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = phi3_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = phi3_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = phi3_pruner.TaylorImportance(
            group_reduction=args.grouping_strategy, taylor=args.taylor)
    logger.log("Use {} pruner...".format(pruner_type))

    num_layers = len(model.model.layers)
    attn_start = max(0, args.block_attention_layer_start)
    attn_end = min(num_layers, args.block_attention_layer_end)
    mlp_start = max(0, args.block_mlp_layer_start)
    mlp_end = min(num_layers, args.block_mlp_layer_end)

    kwargs = {
        "importance": imp,
        "global_pruning": args.global_pruning,
        "iterative_steps": args.iterative_steps,
        "ch_sparsity": args.pruning_ratio,
        "ignored_layers": [],
        "channel_groups": {},
        "consecutive_groups": {
            layer.self_attn.q_proj: layer.self_attn.head_dim
            for layer in model.model.layers
        },
        "customized_pruners": {
            Phi3RMSNorm: phi3_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": (
            [model.model.layers[i].self_attn.q_proj for i in range(attn_start, attn_end)]
            + [model.model.layers[i].mlp.gate_proj for i in range(mlp_start, mlp_end)]
        ),
    }
    logger.log("Pruning Attention Layers = {}".format(list(range(attn_start, attn_end))))
    logger.log("Pruning MLP Layers = {}".format(list(range(mlp_start, mlp_end))))

    pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
    model.zero_grad()

    logger.log("Start Pruning")
    for it in range(args.iterative_steps):
        if pruner_type == 'taylor':
            logger.log("Start Backwarding in iterative step {}/{} ...".format(
                it + 1, args.iterative_steps))
            example_prompts = get_examples(
                args.dataset, tokenizer, args.num_examples, seq_len=args.seq_len_prune)
            if isinstance(example_prompts, torch.Tensor):
                example_prompts = example_prompts.to(args.device)
            if args.taylor in ['param_mix', 'param_second']:
                for j in range(args.num_examples):
                    loss = model(example_prompts[j:j + 1],
                                 labels=example_prompts[j:j + 1]).loss
                    logger.log("sample {} loss = {:.4f}".format(j, loss.item()))
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        p.grad = p.grad * p.grad / args.num_examples
                        if hasattr(p, 'acc_grad'):
                            p.acc_grad += p.grad
                        else:
                            p.acc_grad = copy.deepcopy(p.grad)
                    model.zero_grad()
                    del loss
            else:
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("batch loss = {:.4f}".format(loss.item()))
                loss.backward()

        pruner.step()

        # Recompute num_heads from the actually-pruned q_proj shape.
        for layer in model.model.layers:
            sa = layer.self_attn
            sa.num_heads = sa.q_proj.weight.data.shape[0] // sa.head_dim
            sa.num_key_value_heads = sa.k_proj.weight.data.shape[0] // sa.head_dim
            sa.num_key_value_groups = sa.num_heads // sa.num_key_value_heads
            sa.hidden_size = sa.num_heads * sa.head_dim

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log("After Iter {}/{}, #parameters: {}".format(
            it + 1, args.iterative_steps, after_pruning_parameters))

    # Clear any lingering grads before re-fusing / saving.
    model.zero_grad()
    for name, p in model.named_parameters():
        p.grad = None
        if hasattr(p, 'acc_grad'):
            del p.acc_grad
    del pruner
    gc.collect()
    torch.cuda.empty_cache()

    logger.log("Re-fusing q/k/v_proj -> qkv_proj and gate/up_proj -> gate_up_proj")
    refuse_phi3(model)

    after_pruning_parameters = sum(p.numel() for p in model.parameters())
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(
        before_pruning_parameters, after_pruning_parameters,
        100.0 * after_pruning_parameters / before_pruning_parameters))

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)
                out = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                logger.log(tokenizer.decode(out[0]))
        logger.log("\n==================Finish================\n")

    try:
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'],
                        args.max_seq_len, device=args.eval_device)
        logger.log("PPL after pruning: {}".format(ppl))
    except Exception as e:
        logger.log("PPL evaluation skipped: {}".format(e))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated() / 1024 / 1024))

    if args.save_model:
        model.half()
        torch.save({'model': model, 'tokenizer': tokenizer},
                   logger.best_checkpoint_path)
        logger.log("Saved pruned model to {}".format(logger.best_checkpoint_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Width pruning for Mini-InternVL (Phi-3 backbone)')

    parser.add_argument('--base_model', type=str,
                        default="OpenGVLab/Mini-InternVL-Chat-4B-V1-5",
                        help='Mini-InternVL HF id or local path')
    parser.add_argument('--save_ckpt_log_name', type=str, default="internvl_prune")
    parser.add_argument('--pruning_ratio', type=float, default=0.25)
    parser.add_argument('--pruner_type', type=str, default='taylor',
                        choices=['random', 'l1', 'l2', 'taylor'])
    parser.add_argument('--dataset', type=str, default='c4',
                        help='calibration dataset: c4 | bookcorpus | alpaca | scienceqa_txt')
    parser.add_argument('--seq_len_prune', type=int, default=128)

    # generation (test_after_train)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_seq_len', type=int, default=128)

    # pruning scope (Phi-3-mini is 32 layers; skip first/last a la Bunny/LLaVA defaults)
    parser.add_argument('--block_attention_layer_start', type=int, default=4)
    parser.add_argument('--block_attention_layer_end', type=int, default=30)
    parser.add_argument('--block_mlp_layer_start', type=int, default=4)
    parser.add_argument('--block_mlp_layer_end', type=int, default=30)

    parser.add_argument('--iterative_steps', type=int, default=1)
    parser.add_argument('--grouping_strategy', type=str, default='sum')
    parser.add_argument('--global_pruning', action='store_true')
    parser.add_argument('--taylor', type=str, default='param_first',
                        choices=['vectorize', 'param_second', 'param_first', 'param_mix'])
    parser.add_argument('--num_examples', type=int, default=10)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_device', type=str, default='cuda')
    parser.add_argument('--test_after_train', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()
    main(args)
