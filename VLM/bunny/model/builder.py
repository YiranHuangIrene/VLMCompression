from bunny.model import *
from bunny.model.language_model.bunny_phi import BunnyDistillationModel
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../LLM-Pruner'))
import warnings
import torch
from copy import deepcopy

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, logging

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

def load_pruned_bunny_model(bunny_model_path, pruned_model_path=None, mm=None, lora=None, device_map="", device="cuda", load_8bit=False, load_4bit=False,  **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
        
    model_type = "phi-2"
    if device_map is not None:
        kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    if model_type == 'phi-1.5' or model_type == 'phi-2':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True, local_files_only=True)
        model = BunnyPhiForCausalLM.from_pretrained(bunny_model_path, local_files_only=True, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'phi-3':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyPhi3ForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'stablelm-2':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True, trust_remote_code=True)
        model = BunnyStableLMForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'qwen1.5-1.8b':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyQwen2ForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'minicpm':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyMiniCPMForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'llama3-8b':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyLlamaForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    if pruned_model_path:
        print("loading pruned model")
        pruned_model = torch.load(pruned_model_path, map_location='cpu')
        model.model.layers = deepcopy(pruned_model['model'].model.layers)
        # For LLM-Pruner, change the number of attn heads
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i
        # For shortGPT, change the number of layers
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i
        del pruned_model
    model.resize_token_embeddings(len(tokenizer))
    if mm:
        mm_path = os.path.join(mm, "mm_projector.bin")
        mm_projector_weights = torch.load(mm_path, map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    if lora:
        non_lora_trainables = torch.load(os.path.join(lora, "non_lora_trainables.bin"), map_location='cpu')
        non_lora_trainables = {(k[18:] if k.startswith('module.base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, lora)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    if model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
    model.half()
    return tokenizer, model


def load_pruned_bunny_model_all(bunny_model_path, pruned_model_path=None, mm=None, lora=None, device_map="", device="cuda", load_8bit=False, load_4bit=False,  **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
        
    model_type = "phi-2"
    if device_map is not None:
        kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    
    if model_type == 'phi-1.5' or model_type == 'phi-2':
        try:
            tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True, trust_remote_code=True)
        except Exception:
            from transformers import CodeGenTokenizerFast
            tokenizer = CodeGenTokenizerFast.from_pretrained(bunny_model_path, use_fast=True, trust_remote_code=True)
        model = BunnyPhiForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    elif model_type == 'phi-3':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyPhi3ForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'stablelm-2':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True, trust_remote_code=True)
        model = BunnyStableLMForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'qwen1.5-1.8b':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyQwen2ForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'minicpm':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyMiniCPMForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    elif model_type == 'llama3-8b':
        tokenizer = AutoTokenizer.from_pretrained(bunny_model_path, use_fast=True)
        model = BunnyLlamaForCausalLM.from_pretrained(bunny_model_path, low_cpu_mem_usage=False, **kwargs)
    if pruned_model_path:
        print("loading pruned model")
        pruned_model = torch.load(pruned_model_path, map_location='cpu')
        model.model.layers = pruned_model['model'].model.layers
         # For LLM-Pruner, change the number of attn heads
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        # For shortGPT, change the number of layers
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i
        del pruned_model
    model.resize_token_embeddings(len(tokenizer))
    if mm:
        mm_path = os.path.join(mm, "mm_projector.bin")
        mm_projector_weights = torch.load(mm_path, map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    if lora:
        non_lora_trainables = torch.load(os.path.join(lora, "non_lora_trainables.bin"), map_location='cpu')
        non_lora_trainables = {(k[18:] if k.startswith('module.base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, lora)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Print the number of parameters in the model
    print(f"Number of parameters in the model: {model.num_parameters()}")
    return tokenizer, model, image_processor, context_len

def load_distillation_model(teacher_model_path, student_model_path, pruned_model_path, mm=None, lora=None, device_map=None, device="cuda",  **kwargs):
    # Load teacher model
    _, teacher_model = load_pruned_bunny_model(teacher_model_path)
    # Load student model and tokenizer
    if device_map is not None:
        kwargs = {"device_map": device_map, **kwargs}
    tokenizer = AutoTokenizer.from_pretrained(student_model_path, use_fast=True, local_files_only=True)
    model = BunnyDistillationModel.from_pretrained(student_model_path, low_cpu_mem_usage=False, local_files_only=True, **kwargs)
    if pruned_model_path:
        print("loading pruned model")
        pruned_model = torch.load(pruned_model_path, map_location='cpu')
        model.model.layers = deepcopy(pruned_model['model'].model.layers)
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i

        del pruned_model

    model.resize_token_embeddings(len(tokenizer))
    if mm:
        mm_path = os.path.join(mm, "mm_projector.bin")
        mm_projector_weights = torch.load(mm_path, map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    if lora:
        non_lora_trainables = torch.load(os.path.join(lora, "non_lora_trainables.bin"), map_location='cpu')
        non_lora_trainables = {(k[18:] if k.startswith('module.base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, lora)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
    
    return tokenizer, model, teacher_model
    
    
def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", **kwargs):
    if model_type not in {'phi-1.5', 'phi-2', 'phi-3', 'stablelm-2', 'qwen1.5-1.8b', 'minicpm', 'llama3-8b'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Load Bunny model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print('Loading Bunny from base model...')
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                            config=lora_cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Bunny weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')

            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        # this may be mm projector only
        print('Loading Bunny from base model...')

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    if getattr(model.config, "unfreeze_vision_tower", False):
        if 'lora' in model_name.lower():
            assert model_base is not None
            vision_non_lora_trainables = {k[19:]: v for k, v in non_lora_trainables.items() if
                                          k.startswith('model.vision_tower.')}
            vision_tower.load_state_dict(vision_non_lora_trainables, strict=False)
        else:
            assert model_base is None
            from safetensors.torch import load_file
            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith('safetensors'):
                    vision_weights.update(
                        {k[19:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
                         k.startswith('model.vision_tower.')})
            vision_tower.load_state_dict(vision_weights, strict=True)

    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor, context_len
