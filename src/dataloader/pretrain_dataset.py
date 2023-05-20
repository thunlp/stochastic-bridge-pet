import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
import h5py
import numpy as np
import random


class PretrainDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])
    
    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -100
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, input_mask, segment_ids,
                masked_lm_labels, next_sentence_labels]

class PretrainDatasetWithNeg(Dataset):
    def __init__(self, input_file, max_pred_length, negative_num, vocab_size):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.negative_num = negative_num
        self.vocab_size = vocab_size
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])
    
    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        negative_labels = np.zeros((masked_lm_labels.shape[0], self.negative_num), dtype=np.int64)
        # for i in range(index):
            # negative_labels[masked_lm_positions[i]] = np.random.choice(list(set(range(self.vocab_size)) - set([masked_lm_labels[masked_lm_positions[i]]])), self.negative_num, replace=False)
        # negative_labels[masked_lm_positions[:index]] = np.random.randint(1997, 29612, (index, self.negative_num))
        negative_cand = np.array([np.random.choice(range(30522 - 1000), self.negative_num, replace=False) for _ in range(index)]) + 1000
        negative_labels[masked_lm_positions[:index]] = negative_cand + (negative_cand >= np.array(masked_lm_labels[masked_lm_positions[:index]][:, None]))
        negative_labels = torch.from_numpy(negative_labels)

        return [input_ids, input_mask, segment_ids,
                masked_lm_labels, next_sentence_labels, negative_labels]

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

def create_pretraining_dataset(input_file, max_pred_length, args):
    worker_init = WorkerInitObj(args.seed + args.local_rank)
    train_data = PretrainDataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.batch_size * args.n_gpu,
                                  num_workers=4, worker_init_fn=worker_init,
                                  pin_memory=True)
    return train_dataloader

def create_pretraining_dataset_with_neg(input_file, max_pred_length, negative_num, vocab_size, args):
    worker_init = WorkerInitObj(args.seed + args.local_rank)
    train_data = PretrainDatasetWithNeg(input_file=input_file, max_pred_length=max_pred_length, negative_num=negative_num, vocab_size=vocab_size)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.batch_size * args.n_gpu,
                                  num_workers=4, worker_init_fn=worker_init,
                                  pin_memory=True)
    return train_dataloader