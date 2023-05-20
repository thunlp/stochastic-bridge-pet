from datetime import datetime
import math
import sys
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import dummy_handler, backward_step
from megatron.utils import report_memory
from megatron.model.vision.knn_monitor import compute_feature_bank

from megatron.training import print_datetime, build_train_valid_test_data_iterators, training_log, save_checkpoint_and_time, get_optimizer_param_scheduler, get_model

def get_forward_backward_func():
    forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def forward_step(forward_step_func,
                 data_iterator,
                 teacher_model,
                 student_model,
                 input_tensor,
                 forward_data_store,
                 collect_non_loss_data=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    args = get_args()
    timers = get_timers()

    timers('forward-compute').start()
    student_unwrapped_model = unwrap_model(
        student_model, (torchDDP, LocalDDP, Float16Module))
    teacher_unwrapped_model = unwrap_model(
        teacher_model, (torchDDP, LocalDDP, Float16Module))

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    student_unwrapped_model.set_input_tensor(input_tensor)
    teacher_unwrapped_model.set_input_tensor(input_tensor)
    output_tensor, loss_func = forward_step_func(data_iterator, teacher_model, student_model)
    if mpu.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / get_num_microbatches()
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    if mpu.is_pipeline_stage_after_split() and \
            args.model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]

def forward_backward_no_pipelining(forward_step_func,
                                   data_iterator, teacher_model,
                                   student_model,
                                   optimizer,
                                   timers,
                                   forward_only,
                                   collect_non_loss_data=False):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(student_model) == 1 and len(teacher_model) == 1
    student_model = student_model[0]
    teacher_model = teacher_model[0]

    context_handler = dummy_handler
    if isinstance(student_model, torchDDP):
        context_handler = student_model.no_sync

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(forward_step_func, data_iterator,
                                         teacher_model, student_model, input_tensor, forward_data_store,
                                         collect_non_loss_data)
            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator,
                                 teacher_model, student_model, input_tensor, forward_data_store,
                                 collect_non_loss_data)
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    return forward_data_store

def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              no_load=False):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult)

    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None and not no_load:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        torch.distributed.barrier()
        timers('load-checkpoint').start()
        args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
        if args.half_layer:
            unwrapped_model = unwrap_model(model,
                                           (torchDDP, LocalDDP, Float16Module))
            unwrapped_model[0].half_layer()
        if args.continue_pretraining:
            unwrapped_model = unwrap_model(model,
                                           (torchDDP, LocalDDP, Float16Module))
            optimizer = get_megatron_optimizer(unwrapped_model, no_wd_decay_cond,
                                            scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        if args.only_train_bridge:
            unwrapped_model = unwrap_model(model,
                                           (torchDDP, LocalDDP, Float16Module))
            unwrapped_model[0].bridge.update_embedding(unwrapped_model[0].word_embeddings_weight().cpu())
            unwrapped_model[0].language_model.requires_grad_(False)
            unwrapped_model[0].lm_head.requires_grad_(False)
            unwrapped_model[0].binary_head.requires_grad_(False)
            unwrapped_model[0].language_model.eval()
            unwrapped_model[0].lm_head.eval()
            unwrapped_model[0].binary_head.eval()            
            optimizer = get_megatron_optimizer(unwrapped_model, no_wd_decay_cond,
                                            scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        torch.distributed.barrier()
        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler

def distill(train_valid_test_dataset_provider,
            teacher_model_provider,
            student_model_provider,
            model_type,
            forward_step_func,
            process_non_loss_data_func=None,
            extra_args_provider=None,
            args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()
    teacher_model, _, __ = setup_model_and_optimizer(teacher_model_provider,
                                                     model_type)
    student_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(student_model_provider,
                                                                              model_type)
    unwrap_model(student_model, (torchDDP, LocalDDP, Float16Module))[0].set_num_layer(args.stu_num_layers)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test-data-iterators-setup').start()
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(student_model))
        ]
        train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          teacher_model, student_model, optimizer, opt_param_scheduler,
                          train_data_iterator, valid_data_iterator,
                          process_non_loss_data_func)
    print_datetime('after training is done')

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, teacher_model, student_model,
                                   iteration, process_non_loss_data_func,
                                   False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, student_model, optimizer, opt_param_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, teacher_model, student_model,
                                   0, process_non_loss_data_func,
                                   True)

def train(forward_step_func, teacher_model, student_model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    if not args.only_train_bridge:
        for model_module in teacher_model:
            model_module.train()
        for model_module in student_model:
            model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    timers('interval-time').start()
    print_datetime('before the start of training step')
    report_memory_flag = True
    while iteration < args.train_iters:
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       teacher_model,
                       student_model,
                       optimizer,
                       opt_param_scheduler)
        iteration += 1
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(student_model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, student_model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, teacher_model, student_model,
                                       iteration, process_non_loss_data_func,
                                       False)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, student_model, optimizer,
                                         opt_param_scheduler)
                print_datetime('exiting program after receiving SIGTERM.')
                sys.exit()

        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, student_model, optimizer,
                                     opt_param_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, student_model, optimizer,
                                             opt_param_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, student_model, optimizer,
                                         opt_param_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()


    return iteration

def train_step(forward_step_func, data_iterator,
               teacher_model, student_model, optimizer, opt_param_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in student_model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, teacher_model, student_model,
        optimizer, timers, forward_only=False)

    # Empty unused memory
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if mpu.get_tensor_model_parallel_world_size() > 1 and \
            args.sequence_parallel:
        grads = []
        for model_module in student_model:
            unwrapped_model = unwrap_model( 
                model_module, (torchDDP, LocalDDP, Float16Module))
            for param in unwrapped_model.parameters():
                if getattr(param, 'sequence_parallel', False):
                    grad = param.main_grad if args.DDP_impl == 'local' else param.grad
                    grads.append(grad.data)
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(
            coalesced, group=mpu.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, _unflatten_dense_tensors(
                coalesced, grads)):
            buf.copy_(synced)

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('backward-params-all-reduce').start()
        for model_module in student_model:
            model_module.allreduce_gradients()
        timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce').start()
    if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            unwrapped_model = student_model[0]
        elif mpu.is_pipeline_last_stage(ignore_virtual=True):
            unwrapped_model = student_model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            unwrapped_model = student_model[0]
        unwrapped_model = unwrap_model(
            unwrapped_model, (torchDDP, LocalDDP, Float16Module))

        if unwrapped_model.share_word_embeddings:
            word_embeddings_weight = unwrapped_model.word_embeddings_weight()
            if args.DDP_impl == 'local':
                grad = word_embeddings_weight.main_grad
            else:
                grad = word_embeddings_weight.grad
            torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())

    # All-reduce position_embeddings grad across first (encoder) and split (decoder) 
    # stages to ensure that position embeddings parameters stay in sync.
    # This should only run for T5 models with pipeline parallelism
    if mpu.is_rank_in_position_embedding_group() and \
            mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.pipeline_model_parallel_split_rank is not None:
        unwrapped_model = student_model[0]
        unwrapped_model = unwrap_model(
            unwrapped_model, (torchDDP, LocalDDP, Float16Module))
        assert args.DDP_impl == 'local', \
            'T5 model is only supported with local DDP mode'
        grad = unwrapped_model.language_model.embedding.position_embeddings.weight.main_grad
        torch.distributed.all_reduce(grad, group=mpu.get_position_embedding_group())
    timers('backward-embedding-all-reduce').stop()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(student_model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)


    # Update parameters.
    timers('optimizer').start()
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(student_model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.update_momentum(args.curr_iteration)


    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, teacher_model, student_model,
                               iteration, process_non_loss_data_func,
                               verbose=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, data_iterator, teacher_model, student_model,
        process_non_loss_data_func, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)

def evaluate(forward_step_func,
             data_iterator,
             teacher_model,
             student_model,
             process_non_loss_data_func,
             verbose=False):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(student_model)

    # Turn on evaluation mode which disables dropout.
    for model_module in student_model:
        model_module.eval()
    for model_module in teacher_model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            loss_dicts = forward_backward_func(
                forward_step_func, data_iterator, teacher_model, student_model, optimizer=None,
                timers=None, forward_only=True)

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func, data_iterator, teacher_model, student_model, optimizer=None,
                timers=None, forward_only=True, collect_non_loss_data=True)

    # Move model back to the train mode.
    if not args.only_train_bridge:
        for model_module in teacher_model:
            model_module.train()
        for model_module in student_model:
            model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict, collected_non_loss_data