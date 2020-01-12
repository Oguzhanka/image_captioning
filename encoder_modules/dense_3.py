import torch
import torch.nn as nn


class Dense3(nn.Module):
    def __init__(self, **kwargs):
        super(Dense3, self).__init__()

        self.layers = nn.Sequential(nn.Linear(kwargs["hidden_size"], 2*kwargs["hidden_size"]),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256)).to(kwargs["device"])
        del kwargs

    def forward(self, features, state):
        merged = torch.cat([features, state], dim=1)
        out = self.layers(merged)
        return out
