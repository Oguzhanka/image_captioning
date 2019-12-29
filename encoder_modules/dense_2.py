import torch
import torch.nn as nn


class Dense2(nn.Module):
    def __init__(self, **kwargs):
        super(Dense2, self).__init__()

        self.state = nn.Sequential(nn.Linear(kwargs["word_length"], 1024),
                                   # nn.Tanh(),
                                   nn.Linear(1024, 2048),
                                   # nn.Tanh(),
                                   nn.Linear(2048, kwargs["word_length"])).to(kwargs["device"])

    def forward(self, state):
        out = self.state(state)
        return out
