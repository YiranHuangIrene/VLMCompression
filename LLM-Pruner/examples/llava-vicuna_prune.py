import os
import gc
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VLM'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5,6,7"
import time
import json
import copy
import random
import argparse
from typing import Tuple
from accelerate import Accelerator,hooks


import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig ,AutoTokenizer
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts
from VLM.llava.model.builder import load_pruned_llava_model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)

    if args.short:
        org_pruning_layer = args.pruned_model_path.split("/")[-2].split("_")[-3]
        log_name = "{}_short_{}-{}_{}".format(args.base_model.split("/")[-1], org_pruning_layer, args.pruning_ratio, args.dataset)
    else:
        if args.pruned_model_path: 
            org_pruning_ratio = "-".join(args.pruned_model_path.split("/")[-2].split("_")[2:-1])
            log_name = "{}_{}-{}_{}".format(args.base_model.split("/")[-1], org_pruning_ratio, args.pruning_ratio, args.dataset)
        else:
            log_name = "{}_{}_{}".format(args.base_model.split("/")[-1], args.pruning_ratio,args.dataset)
        if args.iterative_steps > 1:
            log_name += "_iter_{}".format(args.iterative_steps)
        if args.num_examples != 10:
            log_name += "_{}_samples".format(args.num_examples)
    # if args.seq_len_prune:
    #     log_name += "_seq_{}".format(args.seq_len_prune)
    logger = LoggerWithDepth(
        env_name=log_name, 
        config=args.__dict__,
        root_dir=f"{os.path.join(os.path.dirname(__file__), '../')}/LLMPruner/prune_log",
        setup_sublogger=True
    )
    
    tokenizer= AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
        device_map="auto"
    )
    if args.short:
        if  org_pruning_layer == "10":
            pruned_layer = [27, 28, 25, 29, 26, 23, 24, 21, 22, 30]
        elif org_pruning_layer == "5":
            pruned_layer = [27, 28, 25, 29, 26]
        _, model_lora = load_pruned_llava_model(args.base_model,args.pruned_model_path,lora=args.lora,torch_dtype=torch.float16)
        for layer_idx in sorted(pruned_layer, reverse=True):
            try:
                del model.model.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        state_dict_model = model_lora.model.state_dict()
        state_dict_lm_head = model_lora.lm_head.state_dict()
        model.model.load_state_dict(state_dict_model, strict=False)
        model.lm_head.load_state_dict(state_dict_lm_head, strict=True)
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn.rotary_emb.cos_cached.data = model_lora.model.layers[i].self_attn.rotary_emb.cos_cached.data.to(model.model.layers[i].self_attn.rotary_emb.inv_freq.device)
            model.model.layers[i].self_attn.rotary_emb.sin_cached.data = model_lora.model.layers[i].self_attn.rotary_emb.sin_cached.data.to(model.model.layers[i].self_attn.rotary_emb.inv_freq.device)

    for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i
    model.half() 


    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
    
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
        logger.log("PPL before pruning: {}".format(ppl))

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None, 
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    example_prompts = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.seq_len_prune)                  
                    for j in range(args.num_examples):
                        if args.dataset == "llava":
                            input_embeds = example_prompts[j][0]
                            labels = example_prompts[j][1]
                            loss = model(inputs_embeds=input_embeds, labels=labels).loss
                        else:
                            loss = model(example_prompts[j], labels=example_prompts[j]).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad
                example_prompts = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.seq_len_prune, batch=True)
                torch.cuda.empty_cache()
                if args.dataset == "llava":
                    input_embeds = example_prompts[0]
                    labels = example_prompts[1]
                    attention_mask = example_prompts[2]
                    loss = model(inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask).loss
                else:
                    loss = model(example_prompts, labels=example_prompts).loss   
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner

    elif args.channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                #LlamaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
        }

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()
        
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):
            if pruner_type in ['taylor']:
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                example_prompts = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.seq_len_prune, batch=True).to(args.device)
                if args.dataset == "bunny":
                    input_embeds = example_prompts[0]
                    labels = example_prompts[1]
                    attention_mask = example_prompts[2]
                    loss = model(inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask).loss
                else:
                    loss = model(example_prompts, labels=example_prompts).loss   
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()
        
        del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()
    if args.eval_device != "cpu":
        model.half()

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
    
    print(model)
    for n,m in model.named_modules():
        if hasattr(m,"_hf_hook"):
            hooks.remove_hook_from_module(m)
        m.to = torch.nn.Module.to
        m.cuda = torch.nn.Module.cuda
    if args.save_model:
        model.half()
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="liuhaotian/llava-v1.5-7b", help='base model name, or path to the model weights')
    parser.add_argument('--short', action='store_false', help='whether to use the model pruned by shortGPT')
    parser.add_argument('--lora', type=str, default=None, help='path to LoRA model weights')
    parser.add_argument('--pruned_model_path', default=None,
                        help='optional ShortGPT-pruned checkpoint to overlay before width pruning')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.3, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='taylor', help='pruner type')
    parser.add_argument('--dataset', type=str, default='llava', help='dataset for importance calculation: alpaca, bookcorpus, c4, llava')
    parser.add_argument('--seq_len_prune', type=int, default=128, help='sequence length for pruning')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_false', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=21)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=20)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_false', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
