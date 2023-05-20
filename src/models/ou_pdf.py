import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np

class OUBridgeWithLinear(nn.Module):
    def __init__(self, hidden_dim, embedding, project_dim, ignore_index=-100, train_tail_embedding=False) -> None:
        super().__init__()
        self.train_tail_embedding = train_tail_embedding
        mid_project_layers = []
        last_dim = hidden_dim * 2
        self.ignore_index = ignore_index
        assert len(project_dim) > 0
        for dim in project_dim:
            mid_project_layers.append(nn.Linear(last_dim, dim))
            mid_project_layers.append(nn.GELU())
            last_dim = dim
        self.last_dim = last_dim
        self.mid_project = nn.Sequential(*mid_project_layers[:-1])
        self.tail_embedding = nn.Parameter(torch.empty(embedding.shape[0], last_dim), requires_grad=train_tail_embedding)
        self.tail_embedding.data = torch.tensor(PCA(last_dim).fit_transform(embedding.data - embedding.data.mean(dim=0, keepdim=True)), dtype=torch.float)
        self.tail_embedding.data = self.tail_embedding.data / self.tail_embedding.data.norm(dim=-1, keepdim=True)

    def forward(self, hidden_states, labels, attention_mask, neg_labels=None):
        """
        Parameters
        ----------
        hidden_states: [layer + 1, bsz, seq, dim]
        labels: [bsz, seq]
        neg_labels: [bsz, seq, neg_num]
        
        Returns
        -------
        loss
        """
        layer_num, bsz, seq_len, hidden_dim = hidden_states.shape
        layer_num -= 1
        label_mask = labels != self.ignore_index
        ctx = (hidden_states.float() * attention_mask[None, :, :, None]).sum(dim=2) / attention_mask.sum(dim=-1, keepdim=True).unsqueeze(0)
        ctx = torch.cat([ctx[:, i:i+1].expand(-1, label_mask[i].sum(), -1) for i in range(ctx.shape[1])], dim=1)
        selected_hidden_states = hidden_states.masked_select(label_mask.unsqueeze(0).unsqueeze(-1)).view(layer_num + 1, -1, hidden_dim)     # [layer_num + 1, true_label_num, dim]
        selected_hidden_states = torch.cat([ctx.to(selected_hidden_states), selected_hidden_states], dim=-1)
        mid_states = self.mid_project(selected_hidden_states)      # [layer_num, true_label_num, dim]
        true_labels = labels.masked_select(label_mask)              # [true_label_num]
        tail_states = self.tail_embedding[true_labels].unsqueeze(0)      # [1, true_label_num, dim]
        layer_index = torch.arange(1, layer_num + 2, 1, device=hidden_states.device)    # [layer_num]
        T = torch.tensor([1.], device=hidden_states.device, dtype=torch.float)
        t = layer_index / (layer_num + 2) * T
        T_sinh = torch.sinh(T)[:, None, None]           # [1, 1, 1]
        t_sinh = torch.sinh(t)[:, None, None]           # [layer_num + 1, 1, 1]
        diff_sinh = torch.sinh(T - t)[:, None, None]    # [layer_num + 1, 1, 1]
        var = diff_sinh * t_sinh / T_sinh
        residual = (mid_states.float() - t_sinh / T_sinh * tail_states)
        likelihood_loss = (residual * residual).sum(dim=-1, keepdim=True) / (2 * var)
        likelihood_loss = likelihood_loss.mean()
        return likelihood_loss, mid_states

    def update_embedding(self, embedding):
        if self.train_tail_embedding:
            self.tail_embedding.requires_grad_(True)
            self.mid_project.requires_grad_(False)
        else:
            device = self.tail_embedding.device
            self.tail_embedding.data = torch.tensor(PCA(self.last_dim).fit_transform(embedding.data - embedding.data.mean(dim=0, keepdim=True)), dtype=torch.float, device=device)
            self.tail_embedding.data = self.tail_embedding.data / self.tail_embedding.data.norm(dim=-1, keepdim=True)
