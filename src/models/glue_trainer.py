import os
import torch
import torch.nn as nn
import logging
import numpy as np
from transformers import Trainer
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils import WEIGHTS_NAME, is_sagemaker_mp_enabled, WEIGHTS_INDEX_NAME, is_torch_tpu_available
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend, ShardedDDPOption
from transformers.trainer_pt_utils import reissue_pt_warnings
import random
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
from transformers.deepspeed import deepspeed_init
from typing import Optional, List, Dict, Union, Any
from collections import OrderedDict
import warnings
try:
    from apex import amp
except:
    pass
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

class MyTrainer(Trainer):
    def __init__(self, bridge_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge_weight = bridge_weight

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        bert_loss, likelihood_loss, logits, __, ___ = model(**inputs)
        loss = bert_loss + self.bridge_weight * likelihood_loss
        return (loss, (loss, logits[:, 0:1])) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # torch.save(self.model.prompt, os.path.join(output_dir, 'prompt.pt'))
        state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            if 'prompt' in k:
                state_dict[k] = v
        if self.model.bert.config.apply_lora:
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    state_dict[k] = v
        elif self.model.bert.config.apply_adapter:
            for k, v in self.model.state_dict().items():
                if 'adapter' in k:
                    state_dict[k] = v
        else:
            for k, v in self.model.state_dict().items():
                if k[-5:] == '.bias' and v.requires_grad:
                    state_dict[k] = v
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if os.path.exists(best_model_path):
            if self.deepspeed:

                if self.model_wrapped is not None:
                    # this removes the pre-hooks from the previous engine
                    self.model_wrapped.destroy()
                    self.model_wrapped = None

                # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
                deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                    self,
                    num_training_steps=self.args.max_steps,
                    resume_from_checkpoint=self.state.best_model_checkpoint,
                )
                self.model = deepspeed_engine.module
                self.model_wrapped = deepspeed_engine
                self.deepspeed = deepspeed_engine
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                        # If the 'user_content.pt' file exists, load with the new smp api.
                        # Checkpoint must have been saved with the new smp api.
                        smp.resume_from_checkpoint(
                            path=self.state.best_model_checkpoint,
                            tag=WEIGHTS_NAME,
                            partial=False,
                            load_optimizer=False,
                        )
                    else:
                        # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                        # Checkpoint must have been saved with the old smp api.
                        state_dict = torch.load(best_model_path, map_location="cpu")
                        state_dict["_smp_is_partial"] = False
                        load_result = model.load_state_dict(state_dict, strict=True)
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.load_state_dict(state_dict, False)
                # if not is_sagemaker_mp_enabled():
                #     self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb

                run_id = wandb.run.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_loss = metrics["eval_loss"]
                self.state.best_model_checkpoint = output_dir
            elif metric_value == self.state.best_metric:
                if metrics["eval_loss"] < self.state.best_loss:
                    self.state.best_metric = metric_value
                    self.state.best_loss = metrics["eval_loss"]
                    self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)