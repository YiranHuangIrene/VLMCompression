# VLMCompression

Code for the paper **"Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study"**
Yiran Huang, Lukas Thede, Massimiliano Mancini, Wenjia Xu, Zeynep Akata.
[arXiv:2507.20749](https://arxiv.org/abs/2507.20749)

We study structural pruning of the **language backbone** of multimodal LLMs and how to recover the lost performance cheaply:

- **Layer-wise pruning** — ShortGPT-style removal of the least important decoder blocks.
- **Width-wise pruning** — LLM-Pruner-style structural pruning of attention heads and MLP channels.
- **Recovery** — supervised fine-tuning, hidden-state distillation, and projector-only training, on as little as 5% of the original training data.
- **Post-training quantization** — 4/8-bit quantization on top of pruning.

Models evaluated: **LLaVA-v1.5-7B**, **Bunny-v1.0-3B**, and **Mini-InternVL-Chat-4B-V1-5**.

---

## Repository layout

```
.
├── LLM-Pruner/                     # Width-wise pruning (fork of horseee/LLM-Pruner)
│   ├── examples/                   # - per-model pruning entry points (Bunny.py, llava-vicuna_prune.py, …)
│   ├── LLMPruner/                  # - pruner library
│   ├── hf_prune.py / post_training.py / generate.py
│   └── scripts/                    # - example shell scripts
│
├── ShortGPT/                       # Layer-wise pruning (fork of sramshetty/ShortGPT)
│   ├── short_gpt/prune.py          # - main entry for VLM layer pruning
│   ├── short_gpt/short_vlm.py      # - block-influence scoring
│   ├── run_bunny.sh / run_llava.sh # - pruning sweeps
│
├── VLM/                            # VLM-specific glue (models, data, training)
│   ├── bunny/                      # Bunny model + train/eval utilities
│   ├── llava/                      # LLaVA model + train/eval utilities
│   ├── InternVL/                   # Mini-InternVL-Chat-4B-V1-5 (SLURM fine-tuning templates)
│   └── quantization/               # PTQ notebook
│
├── generate_simplified.py          # Inference with a width-wise-pruned LLaMA backbone
└── README.md
```

Each of `LLM-Pruner/`, `ShortGPT/`, and `VLM/InternVL/` retains the upstream project's own README with detailed internals.

---

## Setup

```bash
git clone https://github.com/YiranHuangIrene/VLMCompression
cd VLMCompression

conda create -n vlmc python=3.10 -y
conda activate vlmc

# Core deps for pruning / LLM-Pruner
pip install -r LLM-Pruner/requirement.txt

# InternVL-only extras (optional)
pip install -r VLM/InternVL/requirements.txt

# Common
pip install torch torchvision transformers accelerate peft bitsandbytes deepspeed sentencepiece
```

Base models are downloaded from HuggingFace on first use. To point at a shared cache, set `HF_HOME=/path/to/hf_cache` before running anything.

### Datasets

Fine-tuning uses the standard VLM training mixes:

| Backbone     | Dataset                                   | Env var              |
|--------------|-------------------------------------------|----------------------|
| Bunny-3B     | Bunny-v1.0-data (`bunny_695k.json` + images) | `BUNNY_DATA_PATH`    |
| LLaVA-7B     | LLaVA-v1.5 mix-665k                       | `LLAVA_DATA_PATH`    |
| Mini-InternVL | InternVL-Chat-V1-2-SFT-Data              | `INTERN_META_PATH`   |

See the upstream repos (Bunny, LLaVA, InternVL) for exact download instructions.

---

## Running experiments

### 1. Layer-wise pruning (ShortGPT)

```bash
export BUNNY_DATA_PATH=/path/to/Bunny-v1_0-data/finetune
bash ShortGPT/run_bunny.sh

export LLAVA_DATA_PATH=/path/to/llava
bash ShortGPT/run_llava.sh
```

Or call `prune.py` directly:

```bash
python ShortGPT/short_gpt/prune.py \
  --model_name BAAI/Bunny-v1_0-3B \
  --num_examples 50 \
  --n_prune_layers 10 \
  --save_dir ./prune_log/ \
  --device cuda:0
```

Output: `./prune_log/<model>_pruned_<N>_<K>_samples/pruned_model.bin` plus a JSON log of per-layer importances and the indices removed.

### 2. Width-wise pruning (LLM-Pruner)

```bash
cd LLM-Pruner
python hf_prune.py \
  --pruning_ratio 0.25 \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
  --block_attention_layer_start 4 --block_attention_layer_end 30 \
  --pruner_type taylor --taylor param_first \
  --save_ckpt_log_name llama_prune --save_model
```

See `LLM-Pruner/scripts/` for per-model examples (`llama_prune.sh`, `llava_prune.sh`). The LLaVA SLURM example runs `examples/llava-vicuna_prune.py`; set `REPO_ROOT` to the repo root before `sbatch`.

### 3. Recovery fine-tuning

- **Bunny** — `VLM/bunny/train/train_pruned.py` (SFT) / `train.py` (full). Point it at a pruned checkpoint with `--pruned_model_path`.
- **LLaVA** — `VLM/llava/train/train_pruned.py` (or `train_pruned_mem.py` for xformers/flash-attn).
- **Mini-InternVL** — `VLM/InternVL/internvl_chat/shell/internvl1.5/slurm/*.sh` are example SLURM templates (distillation `-dist`, full FT `-ft`, multimodal-only `-mm`). Edit the hardcoded paths (`#SBATCH --output`, `cd`, `--pruned_model_path`, `--output_dir`, `--deepspeed`) to match your cluster before `sbatch`.

Each training script accepts `--pruned_model_path` to swap in the pruned decoder layers on top of the original base model, followed by `--output_dir` for the recovery checkpoint.

### 4. Quantization

See `VLM/quantization/quantization.ipynb` for 4/8-bit post-training quantization on top of pruned models.

### 5. Inference with a pruned model

```bash
python generate_simplified.py \
  --base_model liuhaotian/llava-v1.5-7b \
  --model_path LLM-Pruner/prune_log/llava-v1.5-7b_0.2/pytorch_model.bin \
  --input_text "Tell me a funny joke"
```

---

## Citation

```bibtex
@article{huang2025investigating,
  title   = {Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study},
  author  = {Huang, Yiran and Thede, Lukas and Mancini, Massimiliano and Xu, Wenjia and Akata, Zeynep},
  journal = {arXiv preprint arXiv:2507.20749},
  year    = {2025}
}
```

## Acknowledgements

This codebase builds on:
- [LLM-Pruner](https://github.com/horseee/LLM-Pruner) (Ma et al., 2023)
- [ShortGPT](https://github.com/sramshetty/ShortGPT) (Men et al., 2024)
- [LLaVA](https://github.com/haotian-liu/LLaVA) (Liu et al., 2023)
- [Bunny](https://github.com/BAAI-DCAI/Bunny) (He et al., 2024)
- [InternVL](https://github.com/OpenGVLab/InternVL) (Chen et al., 2024)
