import torch
import torch.nn as nn
import math

class LoRA(nn.Module):
    def __init__(self, hidden_size, r=8, alpha=8):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, r, bias=False)
        self.up_project = nn.Linear(r, hidden_size, bias=False)
        self.alpha = alpha / r
        self.init_parameters()

    def forward(self, hidden_states):
        hidden_states = self.alpha * self.up_project(self.down_project(hidden_states))
        return hidden_states
    
    def init_parameters(self):
        nn.init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_project.weight)
        if self.down_project.bias is not None:
            nn.init.zeros_(self.down_project.bias)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)

class Adapter(nn.Module):
    def __init__(self, hidden_size, r):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, r)
        self.act = nn.GELU()
        self.up_project = nn.Linear(r, hidden_size)
        self.init_parameters()

    def forward(self, hidden_states):
        x = self.up_project(self.act(self.down_project(hidden_states)))
        return x + hidden_states

    def init_parameters(self):
        nn.init.normal_(self.down_project.weight, std=0.01)
        nn.init.normal_(self.up_project.weight, std=0.01)
        if self.down_project.bias is not None:
            nn.init.zeros_(self.down_project.bias)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)