from collections import defaultdict
import os
import csv
import random

label_map = {
    'CoLA': 1, 
    'SST-2': 1, 
    'MRPC': 0, 
    'QQP': 5, 
    'MNLI': -1, 
    'QNLI': 3, 
    'RTE': 3
}

for task in ['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE']:
    for seed in [42, 43, 44, 45, 46]:
        random.seed(seed)
        with open(f'./data/glue_data/{task}/train.tsv', 'r') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=None)
            lines = defaultdict(list)
            for idx, line in enumerate(reader):
                if idx == 0:
                    head = line
                    continue
                label = line[label_map[task]]
                lines[label].append(line)
        for k in lines.keys():
            random.shuffle(lines[k])
        os.makedirs(f'./data/glue_data/{task}/fewshot', exist_ok=True)
        for shot in [4, 8, 16, 32]:
            with open(f'./data/glue_data/{task}/fewshot/train_seed_{seed}_shot_{shot}.tsv', 'w') as f:
                writer = csv.writer(f, delimiter='\t', quotechar=None)
                writer.writerow(head)
                for k in lines.keys():
                    for line in lines[k][:shot]:
                        writer.writerow(line)
            with open(f'./data/glue_data/{task}/fewshot/dev_seed_{seed}_shot_{shot}.tsv', 'w') as f:
                writer = csv.writer(f, delimiter='\t', quotechar=None)
                writer.writerow(head)
                for k in lines.keys():
                    for line in lines[k][shot:int(shot*2)]:
                        writer.writerow(line)