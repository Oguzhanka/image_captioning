import torch
import random
import torch.nn as nn
from encoder_modules.dense_1 import Dense1


class MultiStepParallel(nn.Module):
    def __init__(self, **kwargs):
        super(MultiStepParallel, self).__init__()
        self.device = kwargs["device"]
        self.embedding = kwargs["embedding"]
        self.encoder = Dense1(**kwargs)
        self.input_size = kwargs["word_length"]

        self.state = None
        self.model = None

        self.sequence_length = kwargs["sequence_length"]

    def forward(self, features):
        """

        :param features: (b, f)
        :return: (b, l, w)
        """

        self._init_states(features)
        batch_size = features.shape[0]
        embedded_input = self.embedding["x_START_"].unsqueeze(dim=0).repeat(batch_size, 1).unsqueeze(0)

        words = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            word_vec = self.encoder(features, out)
            embedded_input = self.embedding(word_vec).unsqueeze(dim=0)
            words.append(word_vec.squeeze(dim=0))

        words = torch.stack(words, dim=1)
        return words

    def caption(self, features):
        """

        :param features: (b, f)
        :return: (b, l, w)
        """
        self._init_states(features)
        batch_size = features.shape[0]

        embedded_input = self.embedding["x_START_"].unsqueeze(dim=0).repeat(batch_size, 1).unsqueeze(0)

        sentence = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            word_vec = self.encoder(features, out)
            embedded_input = self.embedding(word_vec).unsqueeze(dim=0)
            vals, idx = torch.sort(word_vec, dim=1, descending=True)
            word = torch.Tensor([[random.choice(idx[i, :3])] for i in range(batch_size)])
            sentence.append(word)

        words = torch.stack(sentence, dim=1)
        return words
