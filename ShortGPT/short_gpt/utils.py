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


