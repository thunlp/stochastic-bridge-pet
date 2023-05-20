from collections import defaultdict
from curses.ascii import isalpha
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import csv
import os
import torch
import sys
import json
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
csv.field_size_limit(sys.maxsize)

def sst2processor(data_path, mode='train'):
    label_map = {
        '0': 0,
        '1': 1
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, 'SST-2', mode + '.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text = line[0]
            label = label_map[line[1]]
            texts.append(text)
            labels.append(label)
    return texts, labels

def mnliprocessor(data_path, mode='train', prefix=False):
    label_map = {
        'neutral': 0,
        'contradiction': 1,
        'entailment': 2
    }
    texts = []
    labels = []
    if mode == 'dev' or mode == 'test':
        file_name = mode + '_matched.tsv'
    else:
        file_name = mode + '.tsv'
    with open(os.path.join(data_path, 'MNLI', file_name), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text1 = line[8]
            text2 = line[9]
            label = label_map[line[-1]]
            if not prefix:
                if text1[-1] not in ['.', '?', '!']:
                    text1 += '.'
                texts.append(text1 + ' ' + text2)
            else:
                texts.append('Hypothesis: ' + text1 + ' Premise: ' + text2)
            labels.append(label)
    return texts, labels

def sst2processor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        '0': tokenizer.convert_tokens_to_ids(["negative"])[0],
        '1': tokenizer.convert_tokens_to_ids(["positive"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, 'SST-2', mode + '.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text = line[0]
            label = label_map[line[1]]
            texts.append(text)
            labels.append(label)
    return texts, labels

def mnliprocessor_wordlabel(data_path, tokenizer, mode='train', prefix=False):
    label_map = {
        'neutral': tokenizer.convert_tokens_to_ids(["maybe"])[0],
        'contradiction': tokenizer.convert_tokens_to_ids(["no"])[0],
        'entailment': tokenizer.convert_tokens_to_ids(["yes"])[0]
    }
    texts = []
    labels = []
    if mode == 'dev' or mode == 'test':
        file_name = mode + '_matched.tsv'
    else:
        file_name = mode + '.tsv'
    with open(os.path.join(data_path, 'MNLI', file_name), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text1 = line[8]
            text2 = line[9]
            label = label_map[line[-1]]
            if not prefix:
                if text1[-1] not in ['.', '?', '!']:
                    text1 += '.'
                texts.append(text1 + ' ' + text2)
            else:
                texts.append('Hypothesis: ' + text1 + ' Premise: ' + text2)
            labels.append(label)
    return texts, labels


def colaprocessor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        '0': tokenizer.convert_tokens_to_ids(["no"])[0],
        '1': tokenizer.convert_tokens_to_ids(["yes"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, 'CoLA', mode + '.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text = line[-1]
            label = label_map[line[1]]
            texts.append(text)
            labels.append(label)
    return texts, labels

def mrpcprocessor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        '0': tokenizer.convert_tokens_to_ids(["different"])[0],
        '1': tokenizer.convert_tokens_to_ids(["same"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, 'MRPC', mode + '.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text1 = line[-2]
            text2 = line[-1]
            label = label_map[line[0]]
            texts.append(text1 + ' ' + text2)
            labels.append(label)
    return texts, labels

def qnliprocessor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        'not_entailment': tokenizer.convert_tokens_to_ids(["no"])[0],
        'entailment': tokenizer.convert_tokens_to_ids(["yes"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, 'QNLI', mode + '.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text1 = line[1]
            text2 = line[2]
            label = label_map[line[3]]
            texts.append(text1 + ' ' + text2)
            labels.append(label)
    return texts, labels

def qqpprocessor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        '0': tokenizer.convert_tokens_to_ids(["different"])[0],
        '1': tokenizer.convert_tokens_to_ids(["same"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, 'QQP', mode + '.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text1 = line[-3]
            text2 = line[-2]
            label = label_map[line[-1]]
            texts.append(text1 + ' ' + text2)
            labels.append(label)
    return texts, labels

def rteprocessor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        'not_entailment': tokenizer.convert_tokens_to_ids(["no"])[0],
        'entailment': tokenizer.convert_tokens_to_ids(["yes"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, 'RTE', mode + '.tsv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            text1 = line[-3]
            text2 = line[-2]
            label = label_map[line[-1]]
            texts.append(text1 + ' ' + text2)
            labels.append(label)
    return texts, labels

def yelp_polarity_processor_wordlabel(data_path, tokenizer, mode='train'):
    texts = []
    labels = []
    label_map = {
        'positive': tokenizer.convert_tokens_to_ids(["positive"])[0],
        'negative': tokenizer.convert_tokens_to_ids(["negative"])[0]
    }
    with open(os.path.join(data_path, mode + '.jsonl'), 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            label = label_map[line['label']]
            texts.append(line['text'].replace('\\n', ' '))
            labels.append(label)
    return texts, labels

def imdb_processor_wordlabel(data_path, tokenizer, mode='train'):
    texts = []
    labels = []
    label_map = {
        'positive': tokenizer.convert_tokens_to_ids(["positive"])[0],
        'negative': tokenizer.convert_tokens_to_ids(["negative"])[0]
    }
    with open(os.path.join(data_path, mode + '.jsonl'), 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            label = label_map[line['label']]
            texts.append(line['text'].replace('<br /><br />', ' '))
            labels.append(label)
    return texts, labels

def rotten_tomatoes_processor_wordlabel(data_path, tokenizer, mode='train'):
    texts = []
    labels = []
    label_map = {
        'positive': tokenizer.convert_tokens_to_ids(["positive"])[0],
        'negative': tokenizer.convert_tokens_to_ids(["negative"])[0]
    }
    with open(os.path.join(data_path, mode + '.jsonl'), 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            label = label_map[line['label']]
            texts.append(line['text'])
            labels.append(label)
    return texts, labels

def nli_processor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        'neutral': tokenizer.convert_tokens_to_ids(["maybe"])[0],
        'contradiction': tokenizer.convert_tokens_to_ids(["no"])[0],
        'entailment': tokenizer.convert_tokens_to_ids(["yes"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, mode + '.jsonl'), 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            label = label_map[line['label']]
            text1 = line['premise']
            text2 = line['hypothesis']
            if text1[-1] not in ['.', '?', '!']:
                text1 += '.'
            texts.append(text1 + ' ' + text2)
            labels.append(label)
    return texts, labels

def sick_processor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        'neutral': tokenizer.convert_tokens_to_ids(["maybe"])[0],
        'contradiction': tokenizer.convert_tokens_to_ids(["no"])[0],
        'entailment': tokenizer.convert_tokens_to_ids(["yes"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, mode + '.jsonl'), 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            label = label_map[line['label']]
            text1 = line['sentence_A']
            text2 = line['sentence_B']
            if text1[-1] not in ['.', '?', '!']:
                text1 += '.'
            texts.append(text1 + ' ' + text2)
            labels.append(label)
    return texts, labels

def paws_processor_wordlabel(data_path, tokenizer, mode='train'):
    label_map = {
        'not equivalent': tokenizer.convert_tokens_to_ids(["different"])[0],
        'equivalent': tokenizer.convert_tokens_to_ids(["same"])[0]
    }
    texts = []
    labels = []
    with open(os.path.join(data_path, mode + '.jsonl'), 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            label = label_map[line['label']]
            text1 = line['sentence1']
            text2 = line['sentence2']
            texts.append(text1 + ' ' + text2)
            labels.append(label)
    return texts, labels

class GlueDataset(Dataset):
    def __init__(self, args, data_path, task_name, max_length, mode='train') -> None:
        super().__init__()
        self.mode = mode
        self.model_type = args.original_model_path.split('/')[-1]
        if task_name == 'MNLI' and 't5' in self.model_type:
            texts, labels = TASK2PROCESSOR[task_name](data_path, mode, prefix=True)
        else:
            texts, labels = TASK2PROCESSOR[task_name](data_path, mode)
        self.tokenizer = AutoTokenizer.from_pretrained('./pretrained_ckpt/%s' % self.model_type)
        if 'bert' in args.original_model_path:
            tokenized = self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True)
            # labels = self.tokenizer.convert_tokens_to_ids(labels)
            self.all_labels = torch.unique(torch.LongTensor(labels))
            self.dataset = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'token_type_ids': tokenized['token_type_ids'] if 'token_type_ids' in tokenized else [0] * len(tokenized['input_ids']),
                'labels': labels
            }
        elif 't5' in args.original_model_path:
            tokenized = self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True)
            # labels = self.tokenizer(labels).input_ids
            self.all_labels = torch.unique(torch.LongTensor(labels)[:, 0])
            self.dataset = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            }
        
    def __getitem__(self, index):
        if 'bert' in self.model_type:
            return {
                'input_ids': torch.LongTensor(self.dataset['input_ids'][index]), 
                'attention_mask': torch.tensor(self.dataset['attention_mask'][index]), 
                'token_type_ids': torch.LongTensor(self.dataset['token_type_ids'][index]), 
                'labels': torch.LongTensor([self.dataset['labels'][index]])
            }
        elif 't5' in self.model_type:
            return {
                'input_ids': torch.LongTensor(self.dataset['input_ids'][index]), 
                'attention_mask': torch.tensor(self.dataset['attention_mask'][index]), 
                'labels': torch.LongTensor([self.dataset['labels'][index]])
            }

    def __len__(self):
        return len(self.dataset['input_ids'])

class GlueDatasetWordLabel(Dataset):
    def __init__(self, args, data_path, task_name, max_length, mode='train') -> None:
        super().__init__()
        self.mode = mode
        self.model_type = args.original_model_path.split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained('./pretrained_ckpt/%s' % self.model_type)
        if task_name == 'MNLI' and 't5' in self.model_type:
            texts, labels = TASK2PROCESSORWORDLABEL[task_name](data_path, self.tokenizer, mode, prefix=True)
        else:
            texts, labels = TASK2PROCESSORWORDLABEL[task_name](data_path, self.tokenizer, mode)
        if 'deberta' in args.original_model_path:
            tokenized = self.tokenizer(texts, max_length=max_length, truncation=True)
            self.dataset = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'token_type_ids': tokenized['token_type_ids'] if 'token_type_ids' in tokenized else [0] * len(tokenized['input_ids']),
                'labels': labels
            }
        elif 'bert' in args.original_model_path or 'megatron' in args.original_model_path:
            tokenized = self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True)
            # labels = self.tokenizer.convert_tokens_to_ids(labels)
            self.all_labels = torch.unique(torch.LongTensor(labels))
            self.dataset = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'token_type_ids': tokenized['token_type_ids'] if 'token_type_ids' in tokenized else [0] * len(tokenized['input_ids']),
                'labels': labels
            }
        elif 't5' in args.original_model_path:
            tokenized = self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True)
            # labels = self.tokenizer(labels).input_ids
            self.all_labels = torch.unique(torch.LongTensor(labels)[:, 0])
            self.dataset = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            }
        
    def __getitem__(self, index):
        if 'bert' in self.model_type or 'megatron' in self.model_type:
            return {
                'input_ids': torch.LongTensor(self.dataset['input_ids'][index]), 
                'attention_mask': torch.tensor(self.dataset['attention_mask'][index]), 
                'token_type_ids': torch.LongTensor(self.dataset['token_type_ids'][index]), 
                'labels': torch.LongTensor([self.dataset['labels'][index]])
            }
        elif 't5' in self.model_type:
            return {
                'input_ids': torch.LongTensor(self.dataset['input_ids'][index]), 
                'attention_mask': torch.tensor(self.dataset['attention_mask'][index]), 
                'labels': torch.LongTensor([self.dataset['labels'][index]])
            }

    def __len__(self):
        return len(self.dataset['labels'])

    @property
    def all_label_ids(self):
        return np.unique(self.dataset['labels'])

class GlueFewshotDataset(GlueDatasetWordLabel):
    def __init__(self, args, data_path, task_name, max_length, mode='train', num_shot=16, seed=0) -> None:
        if mode in ['train', 'dev']:
            super().__init__(args, data_path, task_name, max_length, f'fewshot/{mode}_seed_{seed}_shot_{num_shot}')
        elif mode == 'test':
            super().__init__(args, data_path, task_name, max_length, 'dev')

def glue_collate_fn(batch):
    if 'token_type_ids' in batch[0]:
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch], dim=0),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch], dim=0),
            'token_type_ids': torch.stack([x['token_type_ids'] for x in batch], dim=0),
            'labels': torch.cat([x['labels'] for x in batch], dim=0),
        }
    else:
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch], dim=0),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch], dim=0),
            'labels': torch.cat([x['labels'] for x in batch], dim=0),
        }

TASK2PROCESSOR = {
    'SST-2': sst2processor,
    'MNLI': mnliprocessor
}

TASK2PROCESSORWORDLABEL = {
    'SST-2': sst2processor_wordlabel,
    'MNLI': mnliprocessor_wordlabel,
    'QNLI': qnliprocessor_wordlabel,
    'QQP': qqpprocessor_wordlabel,
    'RTE': rteprocessor_wordlabel,
    'MRPC': mrpcprocessor_wordlabel,
    'CoLA': colaprocessor_wordlabel,
    'yelp_polarity': yelp_polarity_processor_wordlabel,
    'imdb': imdb_processor_wordlabel,
    'rotten_tomatoes': rotten_tomatoes_processor_wordlabel,
    'sick': sick_processor_wordlabel,
    'super_glue_cb': nli_processor_wordlabel,
    'scitail': nli_processor_wordlabel,
    'paws': paws_processor_wordlabel
}