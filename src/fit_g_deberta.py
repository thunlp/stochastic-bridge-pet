import torch
import os
import logging
from models.deberta import BridgeDebertaForPreTraining
from dataloader.utils import build_train_valid_test_data_iterators
import numpy as np
import torch.distributed as dist
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer, DebertaTokenizer, PreTrainedModel
from transformers.modeling_utils import unwrap_model
from transformers.utils import WEIGHTS_NAME
from typing import Optional

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        likelihood_loss, bert_loss = model(**inputs)
        return (likelihood_loss, (None, likelihood_loss, bert_loss)) if return_outputs else likelihood_loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if isinstance(self.model, BridgeDebertaForPreTraining) or isinstance(unwrap_model(self.model), BridgeDebertaForPreTraining):
            model = self.model
            try:
                model = unwrap_model(model)
            except:
                pass
            torch.save(model.bridge.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        elif not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

def compute_metrics(eval_preds):
    (bridge_loss, bert_loss), labels = eval_preds
    return {'bert_loss': np.mean(bert_loss), 'bridge loss': np.mean(bridge_loss)}

def main(training_args: TrainingArguments, my_args):
    set_seed(training_args.seed)
    model = BridgeDebertaForPreTraining(my_args.project_dim, my_args.pretrained_path, my_args.bridge_type)
    training_args.vocab_size = model.bert.config.vocab_size
    tokenizer = DebertaTokenizer.from_pretrained(my_args.pretrained_path)
    total_batch_size = training_args.per_device_train_batch_size * get_world_size()
    train_dataset, dev_dataset, _ = build_train_valid_test_data_iterators(training_args, [my_args.data_path], training_args.max_steps, training_args.eval_steps, my_args.eval_iter, training_args.per_device_train_batch_size, total_batch_size, my_args.model_type)
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    if training_args.do_train:
        trainer.train()
    if training_args.do_eval:
        print(trainer.evaluate())


if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('--project_dim', nargs='+', type=int, default=[1024, 256, 128])
    parser.add_argument('--eval_iter', type=int, default=10)

    parser.add_argument('--bridge_type', type=str, choices=['brown_pdf', 'ou_pdf', 'brown_sde', 'ou_sde'])

    parser.add_argument('--model_type', type=str, default='deberta')
    parser.add_argument('--data_path', type=str, default='./data/pretrain_data/deberta/wiki_bookcorpus_text_sentence')
    parser.add_argument('--pretrained_path', type=str, default='./pretrained_ckpt/deberta-xlarge')

    training_args, my_args = parser.parse_args_into_dataclasses()
    logging.info(training_args)
    logging.info(my_args)

    main(training_args, my_args)

