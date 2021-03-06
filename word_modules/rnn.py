import torch
import torch.nn as nn
from word_modules.multi_step_rnn import MultiStepRNN
from word_modules.multi_step_parallel import MultiStepParallel


class RNN(MultiStepRNN):
    def __init__(self, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.num_layers = kwargs["num_layers"]
        self.model = nn.RNN(input_size=self.embedding.vector_dim,
                            hidden_size=kwargs["hidden_size"],
                            num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        self.state = features.unsqueeze(dim=0).repeat(self.num_layers, 1, 1)


class RNNParallel(MultiStepParallel):
    def __init__(self, **kwargs):
        super(RNNParallel, self).__init__(**kwargs)
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]
        self.model = nn.RNN(input_size=self.embedding.vector_dim,
                            hidden_size=kwargs["hidden_size"],
                            num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        batch_size = features.shape[0]
        self.state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)


rnn_models = {"RNN": RNN,
              "parallel": RNNParallel}
