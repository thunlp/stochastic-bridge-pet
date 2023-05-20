from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
import math
import torchsde
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


class LipSilu(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.act = nn.SiLU(inplace=inplace)

    def forward(self, x):
        return 0.909 * self.act(x)

class BrownianBridgeSDE(torchsde.SDEStratonovich):
    def __init__(self, hidden_dim, embedding, project_dim, ignore_index=-100, adjoint=False, adaptive=False, method="reversible_heun") -> None:
        super().__init__('diagonal')
        encoder_layers = []
        last_dim = project_dim[-1] + hidden_dim + 1
        self.ignore_index = ignore_index
        self.adaptive = adaptive
        self.method = method
        assert len(project_dim) > 0
        for dim in project_dim:
            encoder_layers.append(nn.Linear(last_dim, dim))
            encoder_layers.append(LipSilu())
            last_dim = dim
        encoder_layers = encoder_layers[:-1]
        self.encoder = nn.Sequential(*encoder_layers)
        self.last_dim = last_dim
        self.tail_embedding = nn.Parameter(torch.empty(embedding.shape[0], last_dim), requires_grad=False)
        self.tail_embedding.data = torch.tensor(PCA(last_dim).fit_transform(embedding.data - embedding.data.mean(dim=0, keepdim=True)), dtype=torch.float)
        self.tail_embedding.data = self.tail_embedding.data / self.tail_embedding.data.norm(dim=-1, keepdim=True)
        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        self.ln = nn.LayerNorm(hidden_dim)
        self.init_linear()

    def init_linear(self):
        with torch.no_grad():
            for module in self.encoder:
                if isinstance(module, nn.Linear):
                    module.weight.normal_(std=0.02)
                    if hasattr(module, 'bias'):
                        module.bias.zero_()

    def f(self, t, y):
        time = t.expand(y.shape[0], 1)
        return self.encoder(torch.cat([time, y, self.ln(self.spline.evaluate(t))], dim=-1))

    def h(self, t, y):
        return (self.tail_states - y) / (1 - t)

    def g(self, t, y):
        return torch.ones_like(y)

    def forward(self, hidden_states, labels):
        """
        Parameters
        ----------
        hidden_states: [layer + 1, bsz, seq, dim]
        labels: [bsz, seq]
        
        Returns
        -------
        loss
        """
        layer_num, bsz, seq_len, hidden_dim = hidden_states.shape
        layer_num -= 1
        label_mask = labels != self.ignore_index
        self.selected_hidden_states = hidden_states.masked_select(label_mask.unsqueeze(0).unsqueeze(-1)).view(layer_num + 1, -1, hidden_dim)     # [layer_num + 1, true_label_num, dim]
        true_labels = labels.masked_select(label_mask)              # [true_label_num]
        self.tail_states = self.tail_embedding[true_labels]      # [vocab_size, dim]
        layer_index = torch.arange(0, layer_num + 2, 1, device=hidden_states.device)    # [layer_num]
        ts = layer_index / (layer_num + 2)
        y0 = torch.zeros(len(true_labels), self.last_dim, device=hidden_states.device, dtype=torch.float)
        coeffs = natural_cubic_spline_coeffs(ts, torch.cat([torch.zeros(1, true_labels.shape[0], hidden_dim, device=self.selected_hidden_states.device), self.selected_hidden_states], dim=0).transpose(0, 1))
        self.spline = NaturalCubicSpline(coeffs)
        zs, kl = self.sdeint_fn(sde=self, y0=y0, ts=ts, method=self.method, adaptive=self.adaptive, logqp=True, dt_min=1./((layer_num + 2)*10), dt=1e-2)
        kl_loss = kl.sum(dim=0).mean()
        return kl_loss, zs

    def update_embedding(self, embedding):
        device = self.tail_embedding.device
        self.tail_embedding.data = torch.tensor(PCA(self.last_dim).fit_transform(embedding.data - embedding.data.mean(dim=0, keepdim=True)), dtype=torch.float, device=device)
        self.tail_embedding.data = self.tail_embedding.data / self.tail_embedding.data.norm(dim=-1, keepdim=True)

