from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from metrics import *


class ShortVLM():

    def __init__(self, model_name, model, tokenizer, layers_path: str, n_prune_layers: Optional[int] = None, device = "cuda"):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.layers = self.model.model.layers
        except AttributeError:
            self.layers = self.model.language_model.model.layers
        self.n_prune_layers = n_prune_layers
        self.importances = [0 for _ in self.layers]  # layer-wise importance scores
        self.device = device

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = [],
        angular: Optional[bool] = False
    ):
        if angular:
            assert self.importances, "Need to compute importances with eval_importance()"
            assert self.n_prune_layers, "Need number of layers to prune, set `n_prune_layers`"
            start_layer = np.argsort(np.array(self.importances[:-self.n_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not layers_to_remove and self.n_prune_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor]):
        n = 1

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i + n]
            self.importances[i] += block_influence(
                in_hidden,
                out_hidden
            ).sum().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        dataloader: torch.utils.data.DataLoader
    ):
        if self.model_name == "OpenGVLab/Mini-InternVL-Chat-4B-V1-5":
            for batch in tqdm(dataloader):
                pixel_values = batch["pixel_values"].half()
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                image_flags = batch["image_flags"]
                position_ids = batch["position_ids"]
                attn_mask = batch["attention_mask"]
                outputs = self.model(
                    pixel_values=pixel_values.to(self.device),
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn_mask.to(self.device),
                    labels=labels.to(self.device),
                    image_flags=image_flags.to(self.device),
                    position_ids=position_ids.to(self.device),
                    output_hidden_states=True,
                )
                self.compute_bi(outputs.hidden_states)
        else:
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attn_mask = batch["attention_mask"]
                images = batch['images']

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                    images=images,
                    output_hidden_states=True,
                )
                self.compute_bi(outputs.hidden_states)

        return
    
    
    
        


        