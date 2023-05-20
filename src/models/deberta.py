from .huggingface_deberta import DebertaForMaskedLM
import torch
import torch.nn as nn
from .brown_pdf import BrownianBridgeWithLinear
from .ou_pdf import OUBridgeWithLinear
from .brown_sde import BrownianBridgeSDE
from .ou_sde import OUBridgeSDE
from typing import Optional


class BridgeDebertaForPreTraining(nn.Module):
    def __init__(self, project_dim, pretrained_path, bridge_type='brown', adjoint=False, adaptive=False, method="reversible_heun"):
        super().__init__()
        self.bert = DebertaForMaskedLM.from_pretrained(pretrained_path)
        self.criterion = nn.CrossEntropyLoss()
        self.bert.requires_grad_(False)
        self.bert.eval()
        self.bridge_type = bridge_type
        if bridge_type == 'brown_pdf':
            self.bridge = BrownianBridgeWithLinear(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim)
        elif bridge_type == 'ou_pdf':
            self.bridge = OUBridgeWithLinear(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim)
        elif bridge_type == 'brown_sde':
            self.bridge = BrownianBridgeSDE(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim, adjoint=adjoint, adaptive=adaptive, method=method)
        elif bridge_type == 'ou_sde':
            self.bridge = OUBridgeSDE(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim, adjoint=adjoint, adaptive=adaptive, method=method)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=0)   # [layer+1, bsz, seq, dim]
        if 'pdf' in self.bridge_type:
            likelihood_loss, mid_states = self.bridge(hidden_states, labels, attention_mask)
        elif 'sde' in self.bridge_type:
            likelihood_loss, mid_states = self.bridge(hidden_states, labels)
        return likelihood_loss, outputs.loss

class DebertaForPromptBase(nn.Module):
    def __init__(self, prompt_length, pretrained_path, apply_lora=False, apply_adapter=False, apply_bias=False, lora_r=8, adapter_r=8) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.bert = DebertaForMaskedLM.from_pretrained(pretrained_path, apply_lora=apply_lora, apply_adapter=apply_adapter, lora_r=lora_r, adapter_r=adapter_r)
        
        if prompt_length > 0:
            self.prompt = nn.Parameter(torch.zeros(prompt_length, self.bert.config.hidden_size))
            with torch.no_grad():
                self.prompt.data[:] = self.bert.get_input_embeddings().weight[-1].unsqueeze(0)
        self.bert.requires_grad_(False)
        self.bert.eval()
        if apply_lora:
            for name, module in self.bert.named_modules():
                if 'lora' in name:
                    module.requires_grad_(True)
        if apply_adapter:
            for name, module in self.bert.named_modules():
                if 'adapter' in name:
                    module.requires_grad_(True)
        if apply_bias:
            for name, parameter in self.bert.named_parameters():
                if '.bias' == name[-5:]:
                    parameter.requires_grad_(True)

    def preprocess_data(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        if inputs_embeds is None:
            bsz, seq_length = input_ids.shape
            inputs_embeds = self.bert.get_input_embeddings()(input_ids)
        else:
            bsz, seq_length = inputs_embeds.shape[:2]
        mask_emb = self.bert.get_input_embeddings().weight[-1].unsqueeze(0).expand(bsz, 1, -1)
        if self.prompt_length > 0:
            prompt = self.prompt.unsqueeze(0).expand(bsz, -1, -1)
            inputs_embeds = torch.cat([mask_emb, prompt, inputs_embeds], dim=1)
        else:
            inputs_embeds = torch.cat([mask_emb, inputs_embeds], dim=1)

        real_length = attention_mask.sum(dim=1, keepdim=True)
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, 1, device=inputs_embeds.device).unsqueeze(0).expand(bsz, -1)
        prompt_length = torch.arange(0, self.prompt_length + 1, 1, device=inputs_embeds.device).unsqueeze(0).expand(bsz, -1) + real_length
        position_ids = torch.cat([prompt_length, position_ids], dim=1).long()

        prompt_mask = torch.ones(bsz, self.prompt_length + 1, device=inputs_embeds.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        prompt_token_type = torch.ones(bsz, self.prompt_length + 1, device=inputs_embeds.device)
        token_type_ids = torch.cat([prompt_token_type, token_type_ids], dim=1).long()

        labels = labels.unsqueeze(1)
        pad_label = -torch.ones(bsz, seq_length + self.prompt_length, device=inputs_embeds.device) * 100
        labels = torch.cat([labels, pad_label], dim=1).long()

        return input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        raise NotImplementedError

    def predict(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        raise NotImplementedError

class DebertaForPrompt(DebertaForPromptBase):
    def __init__(self, prompt_length, pretrained_path, apply_lora=False, apply_adapter=False, apply_bias=False, lora_r=8, adapter_r=8) -> None:
        super().__init__(prompt_length, pretrained_path, apply_lora=apply_lora, apply_adapter=apply_adapter, apply_bias=apply_bias, lora_r=lora_r, adapter_r=adapter_r)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        predict: Optional[bool] = None,
    ):
        input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels = self.preprocess_data(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels)
        outputs = self.bert(
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds, 
            labels=labels,
            output_attentions=output_attentions, 
            output_hidden_states=True, 
            return_dict=return_dict
        )
        return outputs.loss, torch.zeros(1, device=outputs.loss.device).squeeze(), outputs.logits, outputs.hidden_states, None

class BridgeDebertaForPrompt(DebertaForPromptBase):
    def __init__(self, prompt_length, pretrained_path, project_dim, bridge_type, apply_lora=False, apply_adapter=False, apply_bias=False, lora_r=8, adapter_r=8, adjoint=False, adaptive=False, method="reversible_heun") -> None:
        super().__init__(prompt_length, pretrained_path, apply_lora=apply_lora, apply_adapter=apply_adapter, apply_bias=apply_bias, lora_r=lora_r, adapter_r=adapter_r)
        self.criterion = nn.CrossEntropyLoss()
        self.bridge_type = bridge_type
        if bridge_type == 'brown_pdf':
            self.bridge = BrownianBridgeWithLinear(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim)
        elif bridge_type == 'ou_pdf':
            self.bridge = OUBridgeWithLinear(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim)
        elif bridge_type == 'brown_sde':
            self.bridge = BrownianBridgeSDE(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim, adjoint=adjoint, adaptive=adaptive, method=method)
        elif bridge_type == 'ou_sde':
            self.bridge = OUBridgeSDE(self.bert.config.hidden_size, self.bert.get_input_embeddings().weight, project_dim, adjoint=adjoint, adaptive=adaptive, method=method)

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        predict: Optional[bool] = None,
    ):
        input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels = self.preprocess_data(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels)
        outputs = self.bert(
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds, 
            labels=labels,
            output_attentions=output_attentions, 
            output_hidden_states=True, 
            return_dict=return_dict
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=0)   # [layer, bsz, seq, dim]
        if 'pdf' in self.bridge_type:
            likelihood_loss, mid_states = self.bridge(hidden_states, labels, attention_mask)
        elif 'sde' in self.bridge_type:
            likelihood_loss, mid_states = self.bridge(hidden_states, labels)
        return outputs.loss, likelihood_loss, outputs.logits, outputs.hidden_states, mid_states
