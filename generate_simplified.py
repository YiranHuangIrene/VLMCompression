import argparse
import torch
import sys
sys.path.append('/shared-local/aoq609/VLMCompression/LLM-Pruner/')
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM
from VLM.llava.model.language_model.hf_llama.modeling_llama import LlamaForCausalLM
from VLM.llava.model.language_model.hf_llama.configuration_llama import LlamaConfig
PRUNED_CKPT = "/shared-local/aoq609/VLMCompression/LLM-Pruner/LLMPruner/prune_log/{}_{}/pytorch_model.bin"
PRUNED_CFG = "/shared-local/aoq609/VLMCompression/config/{}-{}.json"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def load_model_torch(args):
    pruned_dict = torch.load(args.model_path, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return tokenizer, model

def load_model_state_dict(args):
    state_dict = torch.load(args.model_path)['model'].state_dict()
    weights = {k: v.to(torch.float16) for k, v in state_dict.items()}
    return weights
    

def load_model_hf(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # cfg = LlamaConfig.from_pretrained(args.model_config_path, trust_remote_code=True)
    model = HFLlamaForCausalLM.from_pretrained(
        '/shared-local/aoq609/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965'
    )
    # # weights = load_model_state_dict(args)
    _, model_pruned = load_model_torch(args)
    # Replace per sublayer
    # n_layers = len(model_pruned.model.layers)
    # for i in range(n_layers):
    #     model.model.layers[i].self_attn.q_proj.weight = model_pruned.model.layers[i].self_attn.q_proj.weight
    #     model.model.layers[i].self_attn.q_proj.out_features = model_pruned.model.layers[i].self_attn.q_proj.out_features
    #     model.model.layers[i].self_attn.k_proj.weight = model_pruned.model.layers[i].self_attn.k_proj.weight
    #     model.model.layers[i].self_attn.k_proj.out_features = model_pruned.model.layers[i].self_attn.k_proj.out_features
    #     model.model.layers[i].self_attn.v_proj.weight = model_pruned.model.layers[i].self_attn.v_proj.weight
    #     model.model.layers[i].self_attn.v_proj.out_features = model_pruned.model.layers[i].self_attn.v_proj.out_features
    #     model.model.layers[i].self_attn.o_proj.weight = model_pruned.model.layers[i].self_attn.o_proj.weight
    #     model.model.layers[i].self_attn.o_proj.out_features = model_pruned.model.layers[i].self_attn.o_proj.out_features
    #     model.model.layers[i].mlp.gate_proj.weight = model_pruned.model.layers[i].mlp.gate_proj.weight
    #     model.model.layers[i].mlp.gate_proj.out_features = model_pruned.model.layers[i].mlp.gate_proj.out_features
    #     model.model.layers[i].mlp.down_proj.weight = model_pruned.model.layers[i].mlp.down_proj.weight
    #     model.model.layers[i].mlp.down_proj.out_features = model_pruned.model.layers[i].mlp.down_proj.out_features
    #     model.model.layers[i].mlp.up_proj.weight = model_pruned.model.layers[i].mlp.up_proj.weight
    #     model.model.layers[i].mlp.up_proj.out_features = model_pruned.model.layers[i].mlp.up_proj.out_features
    # Replace the model.model(work)
    # model.model = model_pruned.model
    # Replace the model.model.layers (work)
    model.model.layers = model_pruned.model.layers

    return tokenizer, model

def main(args):
    if args.load_method == "torch":
        tokenizer, model = load_model_torch(args)
    else:
        tokenizer, model = load_model_hf(args)
    
    if device == "cuda":
        model.half()
        model = model.cuda()
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    def evaluate(
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                top_p=top_p,
                temperature=temperature,
                max_length=max_new_tokens,
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output

    output_text = evaluate(args.input_text)
    print(output_text)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="liuhaotian/llava-v1.5-7b", help='base model name')
    parser.add_argument('--input_text', type=str, default='Tell me a funny joke', help = 'Text input for model evaluation')
    parser.add_argument('--prune_ratio', type=float, default=0.2)
    parser.add_argument('--load_method', type=str, default="hf", help="torch or hf")

    args = parser.parse_args()
    args.model_path = PRUNED_CKPT.format(args.base_model.split("/")[-1], args.prune_ratio)
    args.model_config_path = PRUNED_CFG.format(args.base_model.split("/")[-1], args.prune_ratio)
    main(args)


