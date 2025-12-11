import torch
import torch.nn as nn


class MlpPolicyNetwork(nn.Module):


    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.readout = nn.Linear(128, output_dim)


    def forward(self, x):
        act1 = torch.tanh(self.fc1(x))
        act2 = torch.tanh(self.fc2(act1))
        logits = self.readout(act2)
        return logits