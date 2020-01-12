import torch
import torch.nn as nn
from encoder_modules.dense_3 import Dense3


class MultiStepPadded(nn.Module):
    def __init__(self, **kwargs):
        super(MultiStepPadded, self).__init__()
        self.device = kwargs["device"]
        self.embedding = kwargs["embedding"]
        self.encoder = Dense3(**kwargs)
        self.input_size = kwargs["word_length"]

        self.state = None
        self.model = None

        self.sequence_length = kwargs["sequence_length"]

        self.feature_encoder = nn.Sequential(nn.Linear(kwargs["feature_dim"], kwargs["hidden_size"]),
                                             nn.ReLU()).to(self.device)

    def forward(self, features):
        """

        :param features: (b, f)
        :return: (b, l, w)
        """
        self._init_states(features)
        batch_size = features.shape[0]

        im_features = self.feature_encoder(features)

        embedded_input = self.embedding["x_START_"].unsqueeze(dim=0).repeat(batch_size, 1).unsqueeze(0)
        padded_input = torch.zeros(self.sequence_length, embedded_input.shape[1])

        words = []
        for l in range(self.sequence_length):
            padded_input[l] = embedded_input.clone()
            out, self.state = self.model.multi_forward(padded_input, self.state)
            word_vec = self.encoder(im_features, out)
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
            word = torch.argmax(word_vec, dim=1)
            sentence.append(word)

        words = torch.stack(sentence, dim=1)
        return words
