from collections import OrderedDict
import os
from dataloader.glue_dataloader import GlueDatasetWordLabel, glue_collate_fn, GlueFewshotDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch.optim import AdamW
import torch
import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from models.bert import BertForPrompt, BridgeBertForPrompt

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def wandb_log(args, data, step=None):
    if not args.disable_wandb:
        wandb.log(data, step=step)

def train(args, model, optimizer, train_dataloader, dev_dataloader, all_label_ids=None):
    global_step = 0
    scaler = torch.cuda.amp.GradScaler()
    best_val = -1
    best_dev_loss = 1e10
    while True:
        for batch in train_dataloader:
            if global_step == args.training_step:
                return
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            with torch.cuda.amp.autocast():
                bert_loss, likelihood_loss, _, __, ___ = model(**batch)
                loss = bert_loss + args.bridge_weight * likelihood_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step % args.log_step == 0:
                wandb_log(args, {"Classification loss": bert_loss.item(), "Likelihood loss": likelihood_loss.item()}, step=global_step)
                logging.info("Step: %d, Classification Loss: %f, Likelihood Loss: %f" % (global_step, bert_loss.item(), likelihood_loss.item()))
            if global_step % args.eval_step == 0:
                metric, bert_loss, likelihood_loss = eval(args, model, dev_dataloader, all_label_ids)
                logging.info("Eval %s: %f, Classification Loss: %f, Likelihood Loss: %f" % (metric[0], metric[1], bert_loss, likelihood_loss))
                wandb_log(args, {"Eval metric": metric[1], "Eval loss": bert_loss, "Eval Likelihood loss": likelihood_loss}, step=global_step)
                if metric[1] >= best_val:
                    save_flag = False
                    if metric[1] > best_val:
                        best_val = metric[1]
                        best_dev_loss = bert_loss
                        save_flag = True
                    elif bert_loss < best_dev_loss:
                        best_val = metric[1]
                        best_dev_loss = bert_loss
                        save_flag = True
                    if args.save and save_flag:
                        state_dict = OrderedDict()
                        for k, v in model.state_dict().items():
                            if 'prompt' in k:
                                state_dict[k] = v
                            if args.apply_lora and 'lora' in k:
                                state_dict[k] = v
                            if args.apply_adapter and 'adapter' in k:
                                state_dict[k] = v
                            if args.apply_bias and '.bias' == k[-5:]:
                                state_dict[k] = v
                        if len(state_dict) == 0:
                            assert ValueError("State dict has no element")
                        torch.save(state_dict, os.path.join(args.save_path, 'best_model.pt'))

def eval(args, model, dev_dataloader, all_label_ids=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_bert_loss = 0
    total_likelihood_loss = 0
    num_dev = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            with torch.cuda.amp.autocast():
                bert_loss, likelihood_loss, logits, _, __ = model(**batch)
                total_bert_loss += bert_loss * logits.shape[0]
                total_likelihood_loss += likelihood_loss * logits.shape[0]
            all_labels.append(batch['labels'].cpu())
            if all_label_ids is None:
                all_preds.append(logits[:, 0].argmax(dim=-1).cpu())
            else:
                label_logits = logits[:, 0][:, all_label_ids]
                all_preds.append(all_label_ids[label_logits.argmax(dim=-1)].cpu())
            num_dev += logits.shape[0]
    metric = calculate_metric(torch.cat(all_preds), torch.cat(all_labels), args.task_name)
    total_bert_loss = total_bert_loss / num_dev
    total_likelihood_loss = total_likelihood_loss / num_dev
    return metric, total_bert_loss, total_likelihood_loss

def calculate_metric(preds, labels, task_name):
    best_metric = 0
    if task_name in ['MNLI', 'SST-2', 'QNLI', 'RTE']:
        best_metric = accuracy_score(labels, preds)
        metric = ('Accuracy', best_metric)
    elif task_name in ['MRPC', 'QQP']:
        best_metric = f1_score(labels, preds, average='macro', labels=np.unique(labels))
        metric = ('F1', best_metric)
    elif task_name == 'CoLA':
        best_metric = matthews_corrcoef(labels, preds)
        metric = ('Matthews Correlation', best_metric)
    return metric

def main(args):
    if args.fewshot:
        train_dataset = GlueFewshotDataset(args, args.data_path, args.task_name, args.max_length, mode='train', num_shot=args.num_shot, seed=args.fewshot_seed)
        dev_dataset = GlueFewshotDataset(args, args.data_path, args.task_name, args.max_length, mode='dev', num_shot=args.num_shot, seed=args.fewshot_seed)
        test_dataset = GlueFewshotDataset(args, args.data_path, args.task_name, args.max_length, mode='test')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=glue_collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, collate_fn=glue_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, collate_fn=glue_collate_fn)
    else:
        train_dataset = GlueDatasetWordLabel(args, args.data_path, args.task_name, args.max_length, mode='train')
        dev_dataset = GlueDatasetWordLabel(args, args.data_path, args.task_name, args.max_length, mode='dev')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=glue_collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, collate_fn=glue_collate_fn)
    logging.info("Load %d Train Examples, %d steps/epoch" % (len(train_dataset), len(train_dataloader)))
    logging.info("Load %d Dev Examples" % len(dev_dataset))

    if args.model_type == 'bridge_bert':
        model = BridgeBertForPrompt(args.prompt_length, args.original_model_path, args.project_dim, args.bridge_type, apply_lora=args.apply_lora, lora_r=args.lora_r, apply_adapter=args.apply_adapter, adapter_r=args.adapter_r, apply_bias=args.apply_bias)

        if args.load_bridge_path is not None:
            logging.info("Load bridge weights from %s" % args.load_bridge_path)
            try:
                model.bridge.load_state_dict(torch.load(args.load_bridge_path, 'cpu'))
            except:
                model.bridge.load_state_dict(torch.load(args.load_bridge_path, 'cpu')['model']['bridge'])
            model.bridge.requires_grad_(False)
            model.bridge.eval()
    elif args.model_type == 'bert':
        model = BertForPrompt(args.prompt_length, args.original_model_path, apply_lora=args.apply_lora, lora_r=args.lora_r, apply_adapter=args.apply_adapter, adapter_r=args.adapter_r, apply_bias=args.apply_bias)
    else:
        raise ValueError("Model type not supported: %s" % args.model_type)
    model.to(args.device)
    model.eval()

    set_seed(args.seed)
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    logging.info("Optimizable Parameters:")
    num_train_param = 0
    num_total_param = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(n)
            num_train_param += p.numel()
        num_total_param += p.numel()
    logging.info(f"Total number of parameters: {num_total_param}, trainable: {num_train_param}, ratio: {num_train_param/num_total_param*100}%")

    train(args, model, optimizer, train_dataloader, dev_dataloader, torch.from_numpy(train_dataset.all_label_ids).to(args.device) if args.fewshot else None)

    if args.fewshot:
        model.load_state_dict(torch.load(os.path.join(args.save_path, 'best_model.pt'), 'cpu'), strict=False)
        metric, bert_loss, likelihood_loss = eval(args, model, test_dataloader, torch.from_numpy(train_dataset.all_label_ids).to(args.device))
        logging.info("Test %s: %f, Classification Loss: %f, Likelihood Loss: %f" % (metric[0], metric[1], bert_loss, likelihood_loss))
        wandb_log(args, {"Test metric": metric[1], "Test loss": bert_loss, "Test Likelihood loss": likelihood_loss}, step=0)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, choices=['SST-2', 'CoLA', 'MRPC', 'QQP', 'MNLI', 'RTE', 'QNLI'])
    parser.add_argument('--data_path', type=str, default='./data/glue_data')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--original_model_path', type=str)
    parser.add_argument('--load_bridge_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--bridge_type', type=str, choices=['brown_pdf', 'ou_pdf', 'brown_sde', 'ou_sde'])

    parser.add_argument('--apply_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--apply_adapter', action='store_true')
    parser.add_argument('--adapter_r', type=int, default=8)
    parser.add_argument('--apply_bias', action='store_true')

    parser.add_argument('--fewshot', action='store_true')
    parser.add_argument('--fewshot_seed', type=int, default=42)
    parser.add_argument('--num_shot', type=int, default=8)

    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--prompt_length', type=int, default=0)
    parser.add_argument('--project_dim', type=int, nargs='+', default=[768, 768, 256])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--bridge_weight', type=float, default=1)

    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=2000)
    parser.add_argument('--training_step', type=int, default=50000)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--project', type=str, default='debug')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--group', type=str, default='debug')
    parser.add_argument('--disable_wandb', action='store_true')
    args = parser.parse_args()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    print(args)
    if not args.disable_wandb:
        import wandb
        wandb.init(project=args.project, config=vars(args), name=args.name, group=args.group, mode='disabled' if args.disable_wandb else 'offline')
        wandb.config = vars(args)
    main(args)
