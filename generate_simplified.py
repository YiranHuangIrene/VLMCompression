"""Inference with a width-wise-pruned LLaMA backbone (LLM-Pruner output).

Loads the original base model from HuggingFace and swaps in the pruned
decoder layers from a LLM-Pruner checkpoint.
"""
import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_ROOT, "LLM-Pruner"))

from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_torch(args):
    pruned_dict = torch.load(args.model_path, map_location="cpu")
    return pruned_dict["tokenizer"], pruned_dict["model"]


def load_model_hf(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = HFLlamaForCausalLM.from_pretrained(args.base_model)
    _, model_pruned = load_model_torch(args)
    model.model.layers = model_pruned.model.layers
    return tokenizer, model


def main(args):
    if args.load_method == "torch":
        tokenizer, model = load_model_torch(args)
    else:
        tokenizer, model = load_model_hf(args)

    if device == "cuda":
        model = model.half().cuda()

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    def evaluate(prompt, temperature=0.1, top_p=0.75, max_new_tokens=128):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                top_p=top_p,
                temperature=temperature,
                max_length=max_new_tokens,
                return_dict_in_generate=True,
            )
        return tokenizer.decode(out.sequences[0])

    print(evaluate(args.input_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate with a pruned LLaMA backbone")
    parser.add_argument("--base_model", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="HuggingFace id or local path of the original (unpruned) base model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the LLM-Pruner checkpoint (pytorch_model.bin)")
    parser.add_argument("--input_text", type=str, default="Tell me a funny joke")
    parser.add_argument("--load_method", type=str, default="hf", choices=["hf", "torch"],
                        help="'hf' loads base from HF and swaps pruned layers; 'torch' loads the pruned bundle directly")
    args = parser.parse_args()
    main(args)
