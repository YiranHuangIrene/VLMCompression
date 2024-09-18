import random
import json
import numpy as np
import torch
import os
import pwd
from PIL import Image

from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from datasets import load_dataset


if pwd.getpwuid(os.getuid())[0] == "aoq609":
    BUNNY_DATA_PATH = "/shared-network/Bunny-v1_0-data/finetune/"
    LLAVA_DATA_PATH = "/shared-network/llava/"
elif pwd.getpwuid(os.getuid())[0] == "huang17":
    BUNNY_DATA_PATH = "/p/scratch/taco-vlm/datasets/Bunny-v1_0-data/finetune/"
    LLAVA_DATA_PATH = "/p/scratch/taco-vlm/datasets/llava/"


def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_alpaca(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'tatsu-lab/alpaca', split='train'
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0).to('cuda')

def get_bunny(tokenizer, n_samples, seq_len, batch=False):
    from bunny.util.data_utils import LazySupervisedDataset, DataArguments
    from bunny.model.builder import load_pruned_bunny_model_all
    tokenizer,model,image_processor,_ = load_pruned_bunny_model_all("BAAI/Bunny-v1_0-3B")
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    data_args = DataArguments()
    data_args.image_processor = image_processor
    data_args.data_path = "{}bunny_695k.json".format(BUNNY_DATA_PATH)
    data_args.lazy_preprocess = True
    data_args.image_folder = BUNNY_DATA_PATH + "images"
    data_args.image_aspect_ratio = "pad"
    traindata = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    if not batch:
        embeds = []
        labels = []
        history = []
        for _ in range(n_samples):
            sample = {}
            while True:
                i = random.randint(0, len(traindata) - 1)
                sample['input_ids'] = traindata[i]['input_ids']
                if sample['input_ids'].shape[0] >= seq_len and i not in history and traindata[i]['image'] is not None:
                    history.append(i)
                    break
            sample['input_ids'] = sample['input_ids'].unsqueeze(0).to('cuda')
            sample['images'] = traindata[i]['image'].unsqueeze(0).half().to('cuda')
            sample['labels']  = traindata[i]['labels']
                
            _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels = model.prepare_inputs_labels_for_multimodal(position_ids=None,past_key_values=None,labels=None,attention_mask=None,**sample)
            labels.append(new_labels)
            embeds.append(new_input_embeds)
            labels = new_labels
            del model
            del tokenizer
            del image_processor
            torch.cuda.empty_cache()
            return embeds, labels
    else:
        input_ids = []
        images = []
        labels = []
        history = []
        for _ in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                input_id = traindata[i]['input_ids']
                if input_id.shape[0] <= seq_len and i not in history:
                    history.append(i)
                    break
            input_ids.append(traindata[i]['input_ids'])
            images.append(traindata[i]['image'])
            labels.append(traindata[i]['labels'])
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == tokenizer.eos_token_id] = -300
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
                    labels,
                    batch_first=True,
                    padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        input_ids = input_ids[:tokenizer.model_max_length]
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = tokenizer.eos_token_id
        batch = dict(
            input_ids=input_ids.to('cuda'),
            labels=labels.to('cuda'),
            attention_mask=attention_mask.to('cuda'),
        )
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images).half().to('cuda')
        else:
            batch['images'] = images.half().to('cuda')
        _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels= model.prepare_inputs_labels_for_multimodal(position_ids=None,past_key_values=None,**batch) 
        embeds = new_input_embeds
        labels = new_labels
        del model
        del tokenizer
        del image_processor
        torch.cuda.empty_cache()
        return embeds, labels, attention_mask

def get_llava(tokenizer, n_samples, seq_len, batch):
    from llava.train.train_pruned import LazySupervisedDataset, DataArguments
    from llava.model.builder import load_pruned_llava_model_all
    tokenizer,model,image_processor,_ = load_pruned_llava_model_all("liuhaotian/llava-v1.5-7b")
    tokenizer.pad_token = tokenizer.unk_token

    data_args = DataArguments()
    data_args.lazy_preprocess = True
    data_args.data_path = os.path.join(LLAVA_DATA_PATH, "llava_v1_5_mix665k.json")
    data_args.image_folder = LLAVA_DATA_PATH
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    data_args.image_aspect_ratio = "pad"
    data_args.mm_use_im_start_end  = model.config.mm_use_im_start_end 
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    traindata = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    if not batch:
        embeds = []
        labels = []
        history = []
        for _ in range(n_samples):
            sample = {}
            while True:
                i = random.randint(0, len(traindata) - 1)
                sample['input_ids'] = traindata[i]['input_ids']
                if sample['input_ids'].shape[0] >= seq_len and i not in history and traindata[i]['image'] is not None:
                    history.append(i)
                    break
            sample['input_ids'] = sample['input_ids'].unsqueeze(0).to('cuda')
            sample['images'] = traindata[i]['image'].unsqueeze(0).half().to('cuda')
            sample['labels']  = traindata[i]['labels']
                
            _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels = model.to("cuda").prepare_inputs_labels_for_multimodal(position_ids=None,past_key_values=None, attention_mask=None,**sample)
            labels.append(new_labels)
            embeds.append(new_input_embeds)
            labels = new_labels
            del model
            del tokenizer
            del image_processor
            torch.cuda.empty_cache()
            return embeds, labels
    else:
        input_ids = []
        images = []
        labels = []
        history = []
        print("sampling calibration data")
        for _ in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                input_id = traindata[i]['input_ids']
                if input_id.shape[0] <= seq_len and i not in history and traindata[i]['image'] is not None:
                    history.append(i)
                    break
            input_ids.append(traindata[i]['input_ids'])
            images.append(traindata[i]['image'])
            labels.append(traindata[i]['labels'])
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == tokenizer.eos_token_id] = -300
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
                    labels,
                    batch_first=True,
                    padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        input_ids = input_ids[:tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images).half()
        else:
            batch['images'] = images.half()
        _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels= model.prepare_inputs_labels_for_multimodal(position_ids=None,past_key_values=None,**batch) 
        embeds = new_input_embeds.detach()
        labels = new_labels
        del model
        del tokenizer
        del image_processor
        torch.cuda.empty_cache()
        return embeds, labels, attention_mask
    
def get_scienceqa(tokenizer, n_samples, seq_len):
    
    def sqa_doc_to_text(doc):
        context, question, choices = doc["hint"], doc["question"], doc["choices"]
        len_choices = len(choices)
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        if context:
            context = f"Context: {context}\n"
        post_prompt = "\nAnswer with the option's letter from the given choices directly."
        label = options[doc["answer"]]
        return f"{context}{question}\n{choices_str}{post_prompt}{label}."

    dataset = load_dataset("lmms-lab/ScienceQA", 'ScienceQA-FULL')['validation'][:100]
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            if dataset[i]["image"] is None:
                text = sqa_doc_to_text(dataset[i])
                tokenized_sample = tokenizer(text, return_tensors='pt')
            else:
                continue
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        tokenized_samples.append(tokenized_sample.input_ids[:, :-seq_len])
    return torch.cat(tokenized_samples, dim=0 )
    

def get_examples(dataset, tokenizer, n_samples, batch=False, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == "alpaca":
        return get_alpaca(tokenizer, n_samples, seq_len)
    elif dataset == "scienceqa_txt":
        return get_scienceqa(tokenizer, n_samples, seq_len)
    elif dataset == "bunny":
        return get_bunny(tokenizer, n_samples, seq_len, batch=batch)
    elif dataset == "llava":
        return get_llava(tokenizer, n_samples, seq_len, batch=batch)
    else:
        raise NotImplementedError
