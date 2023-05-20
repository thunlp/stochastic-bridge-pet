from functools import partial
from dataloader.glue_dataloader import GlueDatasetWordLabel, GlueFewshotDataset
import torch
import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from transformers import HfArgumentParser, TrainingArguments, set_seed, DebertaTokenizer, DataCollatorWithPadding
from models.deberta import DebertaForPrompt, BridgeDebertaForPrompt
import torch.distributed as dist
from typing import List, Dict, Union
from models.glue_trainer import MyTrainer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

class Collator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        batch['labels'] = batch['labels'].squeeze(-1)
        return batch

def compute_metrics(eval_preds, task_name):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    if task_name in ['MNLI', 'SST-2', 'QNLI', 'RTE']:
        metric = accuracy_score(labels, preds)
        key = 'Accuracy'
    elif task_name in ['MRPC', 'QQP']:
        metric = f1_score(labels, preds, average='macro', labels=np.unique(labels))
        key = 'F1'
    elif task_name == 'CoLA':
        metric = matthews_corrcoef(labels, preds)
        key = 'MatthewsCorrelation'
    torch.cuda.empty_cache()
    return {key: metric}

def main(training_args, my_args):
    set_seed(training_args.seed)
    if my_args.model_type == 'bridge_deberta':
        model = BridgeDebertaForPrompt(my_args.prompt_length, my_args.original_model_path, my_args.project_dim, my_args.bridge_type, apply_lora=my_args.apply_lora, lora_r=my_args.lora_r, apply_adapter=my_args.apply_adapter, adapter_r=my_args.adapter_r, apply_bias=my_args.apply_bias)
        if my_args.load_bridge_path is not None:
            logging.info("Load bridge weights from %s" % my_args.load_bridge_path)
            ckpt = torch.load(my_args.load_bridge_path, 'cpu')
            model.bridge.load_state_dict(ckpt)
            model.bridge.requires_grad_(False)
            model.bridge.eval()
    elif my_args.model_type == 'deberta':
        model = DebertaForPrompt(my_args.prompt_length, my_args.original_model_path, apply_lora=my_args.apply_lora, lora_r=my_args.lora_r, apply_adapter=my_args.apply_adapter, adapter_r=my_args.adapter_r, apply_bias=my_args.apply_bias)
    training_args.vocab_size = model.bert.config.vocab_size
    tokenizer = DebertaTokenizer.from_pretrained(my_args.original_model_path)
    if my_args.fewshot:
        train_dataset = GlueFewshotDataset(my_args, my_args.data_path, my_args.task_name, my_args.max_length, mode='train', num_shot=my_args.num_shot, seed=my_args.fewshot_seed)
        dev_dataset = GlueFewshotDataset(my_args, my_args.data_path, my_args.task_name, my_args.max_length, mode='dev', num_shot=my_args.num_shot, seed=my_args.fewshot_seed)
        test_dataset = GlueFewshotDataset(my_args, my_args.data_path, my_args.task_name, my_args.max_length, mode='test')
    else:
        train_dataset = train_dataset = GlueDatasetWordLabel(my_args, my_args.data_path, my_args.task_name, my_args.max_length, mode='train')
        dev_dataset = GlueDatasetWordLabel(my_args, my_args.data_path, my_args.task_name, my_args.max_length, mode='dev')

    logging.info("Optimizable Parameters:")
    num_train_param = 0
    num_total_param = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(n)
            num_train_param += p.numel()
        num_total_param += p.numel()
    logging.info(f"Total number of parameters: {num_total_param}, trainable: {num_train_param}, ratio: {num_train_param/num_total_param*100}%")

    trainer = MyTrainer(
        bridge_weight=my_args.bridge_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, task_name=my_args.task_name),
        data_collator=Collator(tokenizer, padding=True, max_length=my_args.max_length, pad_to_multiple_of=8),
    )
    trainer.train()

    if my_args.fewshot:
        _, __, metrics = trainer.predict(test_dataset)
        logger.info(metrics)

if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('--task_name', type=str, required=True, choices=['SST-2', 'CoLA', 'MRPC', 'QQP', 'MNLI', 'RTE', 'QNLI'])
    parser.add_argument('--data_path', type=str, default='./data/glue_data')
    parser.add_argument('--original_model_path', type=str)
    parser.add_argument('--load_bridge_path', type=str)
    parser.add_argument('--load_prompt_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--bridge_type', type=str, choices=['brown_pdf', 'ou_pdf', 'brown_sde', 'ou_sde'])

    parser.add_argument('--apply_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--apply_adapter', action='store_true')
    parser.add_argument('--adapter_r', type=int, default=8)
    parser.add_argument('--apply_bias', action='store_true')

    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--prompt_length', type=int, default=0)
    parser.add_argument('--project_dim', type=int, nargs='+', default=[1024, 256, 128])
    parser.add_argument('--bridge_weight', type=float, default=1)

    parser.add_argument('--fewshot', action='store_true')
    parser.add_argument('--fewshot_seed', type=int, default=42)
    parser.add_argument('--num_shot', type=int, default=8)

    # args = parser.parse_args()
    training_args, my_args = parser.parse_args_into_dataclasses()

    print(training_args, my_args)
    
    main(training_args, my_args)
