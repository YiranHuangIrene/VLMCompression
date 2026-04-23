import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
import torch
from transformers import AutoTokenizer
from modeling_internvl_chat import InternVLChatModel, InternVLChatConfig,DistillInternVLChatModel
from copy import deepcopy

def load_pruned_model(model_path, pruned_model_path=None, mm=None, lora=None, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=True)
    model = InternVLChatModel.from_pretrained(model_path, **kwargs)
    if pruned_model_path:
        print('Loading pruned model')
        pruned_model = torch.load(pruned_model_path, map_location='cpu')
        model.language_model.model.layers = deepcopy(pruned_model['model'].language_model.model.layers)
        for layer in model.language_model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.qkv_proj.weight.data.shape[0] // (3 * layer.self_attn.head_dim)
        # For shortGPT, change the number of layers
        for i, layer in enumerate(model.language_model.model.layers):
            layer.self_attn.layer_idx = i
    print('Loaded pruned model')
    print('Model architecture:', model.language_model.model)
    return tokenizer, model

def load_distill_model(model_path, pruned_model_path=None, mm=None, lora=None, **kwargs):
    tokenizer, teacher = load_pruned_model(model_path, **kwargs)
    for name, param in teacher.named_parameters():
        param.requires_grad = False
    student = DistillInternVLChatModel.from_pretrained(model_path, **kwargs)
    if pruned_model_path:
        pruned_model = torch.load(pruned_model_path, map_location='cpu')
        student.language_model.model.layers = deepcopy(pruned_model['model'].language_model.model.layers)
        for layer in student.language_model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.qkv_proj.weight.data.shape[0] // (3 * layer.self_attn.head_dim)
        for i, layer in enumerate(student.language_model.model.layers):
            layer.self_attn.layer_idx = i
    return tokenizer, student, teacher    

def load_pruned_model_devel(model_path, pruned_model=None, mm=None, lora=None,**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=True)
    model = InternVLChatModel.from_pretrained(model_path, **kwargs)
    if pruned_model:
        print('Loading pruned model')
        pruned_model = torch.load(pruned_model, map_location='cpu')
        model.language_model.model.layers = deepcopy(pruned_model['model'].language_model.model.layers)
        for layer in model.language_model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.qkv_proj.weight.data.shape[0] // (3 * layer.self_attn.head_dim)
        # For shortGPT, change the number of layers
        for i, layer in enumerate(model.language_model.model.layers):
            layer.self_attn.layer_idx = i
    print('Loaded pruned model')
    if mm:
        print('Loading mlp weights')
        mm_weights = {}
        paths = os.listdir(mm)
        for path in paths:
            if 'pytorch_model' in path:
                weight_path = os.path.join(mm, path)
                weight = torch.load(weight_path, map_location='cpu')
                for key, value in weight.items():
                    if key.startswith('mlp1'):
                        mm_weights[key[5:]] = value
        model.mlp1.load_state_dict(mm_weights, strict=True)
        print('Loaded mlp weights')
        if lora:
            print("Warp language model with LORA")
            model.wrap_llm_lora(r=16,lora_alpha=32)
            language_model_weights = {}
            for path in paths:
                if 'pytorch_model-000' in path:
                    weight_path = os.path.join(lora, path)
                    weight = torch.load(weight_path, map_location='cpu')
                    for key, value in weight.items():
                        if 'lora' in key:
                            language_model_weights[key] = value
            print('Loading lora weights')
            model.load_state_dict(language_model_weights, strict=False)
            print('Loaded lora weights')      
            # Merge lora weights
            model.language_model = model.language_model.merge_and_unload()
            print('lora weights merged')
    print('Language Model architecture:', model.language_model.model)
    return tokenizer, model