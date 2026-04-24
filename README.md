# MLLM-Compression (IJCV 2026 & GCPR 2025 Oral)

Official implementation of the paper: **Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study**, accepted at IJCV 2026 (Extended Version) and GCPR 2025 (Oral Presentation).

[Authors:Yiran Huang, Lukas Thede, Massimiliano Mancini, Wenjia Xu, Zeynep Akata.]
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
├── LLM-Pruner/                       # Width-wise pruning (fork of horseee/LLM-Pruner)
│   ├── examples/
│   │   ├── Bunny.py                  # - Phi-2 backbone pruning
│   │   ├── llava-vicuna_prune.py     # - LLaMA backbone pruning
│   │   └── InternVL.py               # - Phi-3 backbone pruning (Mini-InternVL)
│   ├── LLMPruner/                    # - pruner library
│   │   ├── models/hf_phi3/           #   - Phi-3 modeling + fusion helpers
│   │   └── pruner/                   #   - per-architecture pruners
│   ├── hf_prune.py / post_training.py / generate.py
│   └── scripts/                      # - example shell scripts
│
├── ShortGPT/                         # Layer-wise pruning (fork of sramshetty/ShortGPT)
│   ├── short_gpt/prune.py            # - main entry for VLM layer pruning
│   ├── short_gpt/short_vlm.py        # - block-influence scoring
│   └── run_bunny.sh / run_llava.sh   # - pruning sweeps
│
├── VLM/                              # VLM-specific glue (models, data, training)
│   ├── bunny/                        # Bunny model + train/eval utilities
│   ├── llava/                        # LLaVA model + train/eval utilities
│   ├── InternVL/                     # Mini-InternVL-Chat-4B-V1-5 (+ SLURM templates)
│   └── quantization/                 # PTQ notebook
│
├── generate_simplified.py            # Inference with a width-wise-pruned LLaMA backbone
├── requirements.txt                  # Unified dependencies
├── LICENSE / NOTICE                  # Apache-2.0 + upstream attribution
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

# Unified dependencies (supersedes LLM-Pruner/requirement.txt and VLM/InternVL/requirements.txt)
pip install -r requirements.txt
```

Base models are downloaded from HuggingFace on first use. Optional:

```bash
export HF_HOME=/path/to/hf_cache           # shared HuggingFace cache
export REPO_ROOT="$(pwd)"                  # needed by the SLURM templates
export WANDB_API_KEY=<your_key>            # if you want online W&B runs
```

### Datasets

Pruning calibration and recovery fine-tuning expect the standard VLM training mixes:

| Backbone      | Dataset                                      | Env var              |
|---------------|----------------------------------------------|----------------------|
| Bunny-3B      | Bunny-v1.0-data (`bunny_695k.json` + images) | `BUNNY_DATA_PATH`    |
| LLaVA-7B      | LLaVA-v1.5 mix-665k                          | `LLAVA_DATA_PATH`    |
| Mini-InternVL | InternVL-Chat-V1-2-SFT-Data (meta JSON)      | `INTERN_META_PATH`   |

See the upstream repos (Bunny, LLaVA, InternVL) for exact download instructions. For InternVL, edit `VLM/InternVL/internvl_chat/shell/data/internvl_1_2_finetune_custom.json` and replace the `<INTERN_DATA_ROOT>` placeholder with the path to your local InternVL-Chat-V1-2-SFT-Data directory.

---

## Running experiments

### 1. Layer-wise pruning (ShortGPT)

Shell wrappers for Bunny and LLaVA:

```bash
export BUNNY_DATA_PATH=/path/to/Bunny-v1_0-data/finetune
bash ShortGPT/run_bunny.sh

export LLAVA_DATA_PATH=/path/to/llava
bash ShortGPT/run_llava.sh
```

Or call `prune.py` directly for any of the three backbones:

```bash
# Bunny
python ShortGPT/short_gpt/prune.py \
  --model_name BAAI/Bunny-v1_0-3B \
  --num_examples 50 --n_prune_layers 10 \
  --save_dir ./prune_log/ --device cuda:0 \
  --bunny_data_path $BUNNY_DATA_PATH

# Mini-InternVL (Phi-3 backbone)
python ShortGPT/short_gpt/prune.py \
  --model_name OpenGVLab/Mini-InternVL-Chat-4B-V1-5 \
  --num_examples 50 --n_prune_layers 5 \
  --save_dir ./prune_log/ --device cuda:0 \
  --intern_meta_path $INTERN_META_PATH
```

Output: `./prune_log/<model>_pruned_<N>_<K>_samples/pruned_model.bin` plus a JSON log of per-layer importances and the indices removed.

### 2. Width-wise pruning (LLM-Pruner)

Per-model entry points:

```bash
# Bunny (Phi-2)
python LLM-Pruner/examples/Bunny.py --pruning_ratio 0.25 --save_model

# LLaVA (LLaMA)
python LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.25 --save_model

# Mini-InternVL (Phi-3) — new
python LLM-Pruner/examples/InternVL.py --pruning_ratio 0.25 --save_model
```

The Mini-InternVL pruner handles Phi-3's fused `qkv_proj` / `gate_up_proj` projections by temporarily unfusing them into LLaMA-style `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` (see `LLM-Pruner/LLMPruner/models/hf_phi3/fusion.py`), running the validated LLaMA pruning machinery, and then re-fusing before saving. The saved checkpoint is a standard `Phi3ForCausalLM` that can be loaded with stock `from_pretrained` or fed to the InternVL recovery loader.

The low-level `hf_prune.py` script is still available for LLaMA-only workflows:

```bash
python LLM-Pruner/hf_prune.py \
  --pruning_ratio 0.25 --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
  --block_attention_layer_start 4 --block_attention_layer_end 30 \
  --pruner_type taylor --taylor param_first \
  --save_ckpt_log_name llama_prune --save_model
```

Example SLURM templates: `LLM-Pruner/scripts/{llama,llava,internvl}_prune.sh`. All read env vars (`REPO_ROOT`, `WANDB_API_KEY`, dataset paths) from the submitting shell — no hardcoded absolute paths.

### 3. Recovery fine-tuning

- **Bunny** — `VLM/bunny/train/train_pruned.py` (SFT) / `train.py` (full). Pass `--pruned_model_path /path/to/pruned_model.bin`.
- **LLaVA** — `VLM/llava/train/train_pruned.py` (or `train_pruned_mem.py` for xformers/flash-attn).
- **Mini-InternVL** — `VLM/InternVL/internvl_chat/shell/internvl1.5/slurm/*.sh` are example SLURM templates for three recovery modes: distillation (`-dist`), full fine-tuning (`-ft`), and multimodal-only (`-mm`). They expect these env vars in the submitting shell:

  ```bash
  export REPO_ROOT="$(pwd)"
  export WANDB_API_KEY=<your_key>
  export INTERN_META_PATH=/path/to/InternVL-Chat-V1-2-SFT-Data/meta.json
  # optional: export HF_HOME=/path/to/hf_cache
  sbatch VLM/InternVL/internvl_chat/shell/internvl1.5/slurm/Mini-InternVL-Chat-4B-V1-5_pruned_5_50_samples-ft.sh
  ```

Each training script accepts `--pruned_model_path` to swap in the pruned decoder layers on top of the original base model, followed by `--output_dir` for the recovery checkpoint. The InternVL loader at `VLM/InternVL/internvl_chat/internvl/model/internvl_chat/builder.py` auto-detects both the bare-Phi-3 checkpoints produced by `LLM-Pruner/examples/InternVL.py` and the wrapped InternVLChatModel checkpoints from ShortGPT.

### 4. Quantization

See `VLM/quantization/quantization.ipynb` for 4/8-bit post-training quantization on top of pruned models.

### 5. Inference with a pruned model

```bash
python generate_simplified.py \
  --base_model liuhaotian/llava-v1.5-7b \
  --model_path /path/to/pytorch_model.bin \
  --input_text "Tell me a funny joke"
```

---

## License

This repository is released under the Apache License 2.0 (see `LICENSE`). It bundles modified forks of LLM-Pruner, ShortGPT, LLaVA, Bunny, and InternVL; upstream licenses are preserved under each subdirectory and attribution is recorded in `NOTICE`.

## Citation

```bibtex
@inproceedings{huang2025investigating,
  title={Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study},
  author={Huang, Yiran and Thede, Lukas and Mancini, Massimiliano and Xu, Wenjia and Akata, Zeynep},
  booktitle={DAGM German Conference on Pattern Recognition},
  pages={320--336},
  year={2025},
  organization={Springer}
}
```

## Acknowledgements

This codebase builds on:
- [LLM-Pruner](https://github.com/horseee/LLM-Pruner) (Ma et al., 2023)
- [ShortGPT](https://github.com/sramshetty/ShortGPT) (Men et al., 2024)
- [LLaVA](https://github.com/haotian-liu/LLaVA) (Liu et al., 2023)
- [Bunny](https://github.com/BAAI-DCAI/Bunny) (He et al., 2024)
- [InternVL](https://github.com/OpenGVLab/InternVL) (Chen et al., 2024)
