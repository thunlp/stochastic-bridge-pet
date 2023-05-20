# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT model."""

import torch

from megatron import get_args
from megatron import mpu
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model import LayerNorm
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule\

from models.brown_pdf import BrownianBridgeWithLinear
from models.ou_pdf import OUBridgeWithLinear
from models.brown_sde import BrownianBridgeSDE
from models.ou_sde import OUBridgeSDE

def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = (extended_attention_mask < 0.5)

    return extended_attention_mask

def bert_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, hidden_size, init_method,
                 layernorm_epsilon, parallel_output):

        super(BertLMHead, self).__init__()

        args = get_args()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        mpu.set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        setattr(self.dense.weight, 'sequence_parallel', args.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', args.sequence_parallel)

        self.layernorm = LayerNorm(hidden_size,
                                   eps=layernorm_epsilon,
                                   sequence_parallel=args.sequence_parallel)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu
        elif args.onnx_safe:
            self.gelu = erf_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output


def post_language_model_processing(lm_output, pooled_output,
                                   lm_head, binary_head,
                                   lm_labels,
                                   logit_weights,
                                   fp16_lm_cross_entropy):
    # Output.
    lm_logits = lm_head(
        lm_output, logit_weights)

    binary_logits = None
    if binary_head is not None:
        binary_logits = binary_head(pooled_output)

    if lm_labels is None:
        # [s b h] => [b s h]
        return lm_logits.transpose(0,1).contiguous(), binary_logits
    else:
        # [b s] => [s b]
        lm_labels = lm_labels.transpose(0,1).contiguous()
        # lm_logits : [s, b, h] and lm_labels: [s, b]
        if fp16_lm_cross_entropy:
            assert lm_logits.dtype == torch.half
            lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, lm_labels)
        else:
            lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(),
                                                       lm_labels)
        # [s, b] => [b s]
        lm_loss = lm_loss.transpose(0,1).contiguous()
        return lm_loss, binary_logits


class BertModel(MegatronModule):
    """Bert Language model."""

    def __init__(self,
                 num_tokentypes=2,
                 add_binary_head=True,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super(BertModel, self).__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process

        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=self.add_binary_head,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings(init_method_normal)
        if self.post_process:
            self.lm_head = BertLMHead(
                self.word_embeddings_weight().size(0),
                args.hidden_size, init_method, args.layernorm_epsilon, parallel_output)
            self._lm_head_key = 'lm_head'
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(args.hidden_size, 2,
                                                    init_method)
                self._binary_head_key = 'binary_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, bert_model_input, attention_mask,
                tokentype_ids=None, lm_labels=None, return_logits=False):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        input_ids = bert_model_input
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )

        if self.post_process and self.add_binary_head:
            lm_output, pooled_output = lm_output
        else:
            pooled_output = None

        if self.post_process:
            return post_language_model_processing(lm_output, pooled_output,
                                                  self.lm_head, self.binary_head,
                                                  None if return_logits else lm_labels,
                                                  self.word_embeddings_weight(),
                                                  self.fp16_lm_cross_entropy)
        else:
            return lm_output


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        if self.post_process:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.post_process and self.add_binary_head:
            state_dict_[self._binary_head_key] \
                = self.binary_head.state_dict(destination, prefix, keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
    
    def half_layer(self):
        self.language_model.encoder.num_layers = int(self.language_model.encoder.num_layers / 2)
        new_layers = []
        for i in range(self.language_model.encoder.num_layers):
            new_layers.append(self.language_model.encoder.layers[i])
        self.language_model.encoder.layers = torch.nn.ModuleList(new_layers)

    def set_num_layer(self, num_layers):
        self.language_model.encoder.num_layers = num_layers
        new_layers = []
        for i in range(num_layers):
            new_layers.append(self.language_model.encoder.layers[i])
        self.language_model.encoder.layers = torch.nn.ModuleList(new_layers)

class BridgeBertModel(MegatronModule):
    """Bert Language model."""

    def __init__(self,
                 num_tokentypes=2,
                 add_binary_head=True,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 project_dim=[1024, 256, 128],
                 adjoint=False,
                 adaptive=False,
                 method="reversible_heun",
                 bridge_type='brown'):
        super(BridgeBertModel, self).__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.bridge_weight = args.bridge_weight
        self.only_train_bridge = args.only_train_bridge

        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=self.add_binary_head,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings(init_method_normal)
        if self.post_process:
            self.lm_head = BertLMHead(
                self.word_embeddings_weight().size(0),
                args.hidden_size, init_method, args.layernorm_epsilon, parallel_output)
            self._lm_head_key = 'lm_head'
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(args.hidden_size, 2,
                                                    init_method)
                self._binary_head_key = 'binary_head'

        self.bridge_type = bridge_type
        if bridge_type == 'brown_pdf':
            self.bridge = BrownianBridgeWithLinear(self.language_model.hidden_size, self.word_embeddings_weight().cpu(), project_dim, ignore_index=-1)
        elif bridge_type == 'ou_pdf':
            self.bridge = OUBridgeWithLinear(self.language_model.hidden_size, self.word_embeddings_weight().cpu(), project_dim, ignore_index=-1)
        elif bridge_type == 'brown_sde':
            self.bridge = BrownianBridgeSDE(self.language_model.hidden_size, self.word_embeddings_weight().cpu(), project_dim, ignore_index=-1, adjoint=adjoint, adaptive=adaptive, method=method)
        elif bridge_type == 'ou_sde':
            self.bridge = OUBridgeSDE(self.language_model.hidden_size, self.word_embeddings_weight().cpu(), project_dim, ignore_index=-1, adjoint=adjoint, adaptive=adaptive, method=method)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, bert_model_input, attention_mask,
                tokentype_ids=None, lm_labels=None, output_enc_hidden=False):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        input_ids = bert_model_input
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids,
            output_all_layers=True
        )

        if self.post_process and self.add_binary_head:
            lm_output, pooled_output, all_layer_hidden = lm_output
        else:
            pooled_output = None

        if 'pdf' in self.bridge_type:
            brown_loss, _ = self.bridge(torch.stack(all_layer_hidden, dim=0).permute(0, 2, 1, 3), lm_labels, attention_mask)
        elif 'sde' in self.bridge_type:
            brown_loss, _ = self.bridge(torch.stack(all_layer_hidden, dim=0).permute(0, 2, 1, 3), lm_labels)
        brown_loss = brown_loss * self.bridge_weight
        if self.post_process:
            return post_language_model_processing(lm_output, pooled_output,
                                                  self.lm_head, self.binary_head,
                                                  lm_labels,
                                                  self.word_embeddings_weight(),
                                                  self.fp16_lm_cross_entropy) + (brown_loss, )
        else:
            return lm_output


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        if not self.only_train_bridge:
            state_dict_[self._language_model_key] \
                = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
            if self.post_process:
                state_dict_[self._lm_head_key] \
                    = self.lm_head.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
            if self.post_process and self.add_binary_head:
                state_dict_[self._binary_head_key] \
                    = self.binary_head.state_dict(destination, prefix, keep_vars)
            # Save word_embeddings.
            if self.post_process and not self.pre_process:
                state_dict_[self._word_embeddings_for_head_key] \
                    = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_['bridge'] = self.bridge.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
