import os
import pwd
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VLM'))
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import random
import numpy as np
import argparse
from tqdm import tqdm
from utils import Collate
from short_vlm import ShortVLM
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_data(model_name, tokenizer, image_processor, num_samples, device) -> Dataset:
    print("loading dataset")
    if model_name == "BAAI/Bunny-v1_0-3B":
        from bunny.util.data_utils import DataArguments, LazySupervisedDataset
        if tokenizer.unk_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        data_args = DataArguments()
        data_args.data_path = os.path.join(BUNNY_DATA_PATH, "bunny_695k.json")
        data_args.image_folder = os.path.join(BUNNY_DATA_PATH, "images")
        data_args.image_processor = image_processor
        data_args.lazy_preprocess = True
        data_args.image_aspect_ratio = "pad"
        dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                            data_path=data_args.data_path,
                                            data_args=data_args)

    elif model_name == "liuhaotian/llava-v1.5-7b":
        from llava.train.train_pruned import  DataArguments, LazySupervisedDataset
        
        data_args = DataArguments()
        data_args.data_path = os.path.join(LLAVA_DATA_PATH, "llava_v1_5_mix665k.json")
        data_args.image_folder = LLAVA_DATA_PATH
        data_args.is_multimodal = True
        data_args.image_processor = image_processor
        data_args.lazy_preprocess = True
        data_args.image_aspect_ratio = "pad"
        data_args.mm_use_im_start_end = False
        dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                            data_path=data_args.data_path,
                                            data_args=data_args)
        
    if num_samples is not None and num_samples > 0:
        total_samples = len(dataset)
        assert num_samples < total_samples
        indices = random.sample(range(total_samples), num_samples)
        dataset = Subset(dataset, indices)
    print(f"{num_samples} samples loaded")
    collate_fn = Collate(tokenizer, model, device)
    print("loading dataloader")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    print("dataloader loaded")
    return dataset, dataloader

def get_model_tokenizer(model_name, device):
    print("loading model and tokenizer")
    if model_name == "BAAI/Bunny-v1_0-3B":
        from bunny.model.builder import load_pruned_bunny_model_all
        tokenizer, model, image_processor, context_len = load_pruned_bunny_model_all(model_name, device=device, torch_dtype=torch.float16)
    elif model_name == "liuhaotian/llava-v1.5-7b":
        from llava.model.builder import load_pruned_llava_model_all
        tokenizer, model, image_processor, context_len = load_pruned_llava_model_all(model_name, device=device, torch_dtype=torch.float16)
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
    else:
        raise ValueError("Unknown model")
    
    model.to(device)
    print("model and tokenizer loaded")
    return model, tokenizer, image_processor

if __name__ == "__main__":
    if pwd.getpwuid(os.getuid())[0] == "aoq609":
        BUNNY_DATA_PATH = "/shared-network/Bunny-v1_0-data/finetune/"
        LLAVA_DATA_PATH = "/shared-network/llava/"
    elif pwd.getpwuid(os.getuid())[0] == "huang17":
        BUNNY_DATA_PATH = "/p/scratch/taco-vlm/datasets/Bunny-v1_0-data/finetune/"
        LLAVA_DATA_PATH = "/p/scratch/taco-vlm/datasets/llava/"
    parser = argparse.ArgumentParser(description='Layerwise Pruning')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--model_name', type=str, default="BAAI/Bunny-v1_0-3B", help='base model name, or path to the model weights')
    parser.add_argument('--device', type=str, default="cuda:2", help='device to run the model on')
    parser.add_argument('--n_prune_layers', type=int, default=10, help='number of layers to prune')
    parser.add_argument('--num_examples', type=int, default=50, help='number of examples to load')
    parser.add_argument('--save_dir', type=str, default="/shared-local/aoq609/VLMCompression/ShortGPT/prune_log/", help='path to save the pruned model')
    args = parser.parse_args()
    model_name = args.model_name
    device = args.device
    n_prune_layers = args.n_prune_layers
    num_examples = args.num_examples
    save_dir = os.path.join(args.save_dir, f"{model_name.split('/')[-1]}_pruned_{n_prune_layers}_{num_examples}_samples/")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_save_path = os.path.join(save_dir, "pruned_model.bin")
    log_save_path = os.path.join(save_dir, "pruned_model.log")
    set_random_seed(args.seed)
    model, tokenizer, image_processor = get_model_tokenizer(model_name=model_name, device=device)
    dataset, dataloader = get_data(model_name=model_name, tokenizer=tokenizer, image_processor=image_processor, num_samples=num_examples, device=device)
    shortvlm = ShortVLM(model=model, tokenizer=tokenizer, layers_path='model.layers', n_prune_layers=n_prune_layers)
    shortvlm.eval_importance(dataloader=dataloader)
    n_layers = len(shortvlm.importances)
    log = {
        "model": model_name,
        "n_prune_layers": n_prune_layers,
        "num_examples": num_examples,
        "layers_removed": None,
        "importances_before_pruning": {},
    }
    print("Before pruning:")
    for i in range(n_layers):
        log['importances_before_pruning'][f"Layer {i+1}"] = shortvlm.importances[i]
        print(f"Layer {i+1}: {shortvlm.importances[i]}")
    layers_removed = shortvlm.remove_layers()
    log['layers_removed'] = layers_removed
    log['number of parameters after pruning'] = sum(p.numel() for p in model.parameters())
    print(model)
    with open(log_save_path, "w") as f:
        json.dump(log, f, indent=4)
    torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, model_save_path)

    
    
        