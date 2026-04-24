from transformers import Trainer, PreTrainedModel
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_peft_available,
    is_safetensors_available,
    logging,
)
from peft import PeftModel
import torch
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn

TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.get_logger(__name__)
if is_safetensors_available():
    import safetensors.torch
    
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return
    
class InternvlDistillationTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = {}
        out = model(**inputs, output_hidden_states=True)
        outputs['student'] = out[0]
        outputs['teacher'] = out[1]
        # calculate l2 loss on the last layer feature between student and teacher 

        student_last_layer = outputs['student'].hidden_states[-1]
        teacher_last_layer = outputs['teacher'].hidden_states[-1]
        l2_loss = nn.functional.mse_loss(student_last_layer, teacher_last_layer)   
        xe_loss = outputs['student']["loss"]
        loss = self.args.dist_alpha * l2_loss + (1 - self.args.dist_alpha) * xe_loss

        return (loss, outputs) if return_outputs else loss