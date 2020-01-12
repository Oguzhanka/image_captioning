import torch
import torch.nn as nn


class Dense1(nn.Module):
    def __init__(self, **kwargs):
        super(Dense1, self).__init__()

        self.feature = nn.Sequential(nn.Linear(kwargs["feature_dim"], 4096),
                                     nn.ReLU(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(),
                                     nn.Linear(4096, 256)).to(kwargs["device"])
        self.state = nn.Sequential(nn.Linear(kwargs["hidden_size"], 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256)).to(kwargs["device"])

        self.merger = nn.Sequential(nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, kwargs["word_length"]),
                                    nn.ReLU()).to(kwargs["device"])
        del kwargs

    def forward(self, features, state):
        feature_encoded = self.feature(features)
        state_encoded = self.state(state).squeeze(dim=0)
        merged = torch.cat([feature_encoded, state_encoded], dim=1)
        out = self.merger(merged)
        return out
