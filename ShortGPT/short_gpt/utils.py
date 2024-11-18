import os
import sys
import pwd
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VLM'))
import torch
from torch.utils.data import DataLoader
from typing import Dict, Sequence, Optional
import transformers
from llava.constants import IGNORE_INDEX
from bunny.constants import IGNORE_INDEX
import math

import json
class Collate():
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels"))

        if self.model =="BAAI/Bunny-v1_0-3B":
            # Handling EOS token case
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                for input_id in input_ids:
                    input_id[input_id == self.tokenizer.eos_token_id] = -300

        # Padding input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        # Truncate to the tokenizer's model max length
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        # Create attention masks
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Truncate labels as well
        labels = labels[:, :self.tokenizer.model_max_length]

        if self.model =="BAAI/Bunny-v1_0-3B":
            # Reverse EOS token handling if needed
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                for input_id in input_ids:
                    input_id[input_id == -300] = self.tokenizer.eos_token_id

        # Prepare the batch
        batch = dict(
            input_ids=input_ids.to(self.device),
            labels=labels.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )

        # Handle optional 'image' field in instances
        if 'image' in instances[0]:
            images = [instance['image'].to(self.device).half() for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    num_image_token,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
):
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../VLM/InternVL/internvl_chat/internvl/train'))
    from internvl_chat_finetune import LazySupervisedDataset, PackedDataset, WeightedConcatDataset, ConcatDataset
    datasets = []
    lengths = []
    data_rank=0,
    data_world_size=1,
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset
