import torch
import torch.nn as nn
import pandas as pd


class Embedding(nn.Module):
    def __init__(self, **params):
        super(Embedding, self).__init__()

        self.embed_path = params["embed_path"]
        self.load_embed = params["load_embed"]
        self.train_embed = params["train_embed"]
        self.device = params["device"]

        self.num_spec_chars = params["num_spec_chars"]
        self.embed_length = params["embed_length"]

        self.word2vecs = None
        self.__load_embedding()

    def forward(self, input_):
        """

        :param input_: (b, w)
        :return: (b, w_e)
        """
        if self.num_spec_chars == 0:
            return torch.matmul(input_, self.word2vecs)
        return torch.matmul(input_, self.word2vecs[:-self.num_spec_chars])

    def int2vec(self, idx):
        """

        :param idx: (b, l)
        :return: (b, l, w)
        """
        vecs = []
        for b in range(idx.shape[0]):
            vecs.append(torch.stack([self.word2vecs[w].clone() for w in idx[b]], dim=0))
        return torch.stack(vecs, dim=0)

    def word2vec(self, word):
        """

        :param word: (b, l)
        :return: (b, l, w)
        """
        int_sequences = []
        for b in range(word.shape[0]):
            int_sequences.append([int(self.__word2int[idx]) for idx in word[b]])

        int_sequences = torch.Tensor(int_sequences).int()
        return self.int2vec(int_sequences)

    def translate(self, idx):
        """

        :param idx: (b, l)
        :return: (b, l)
        """

        translated = []
        for b in range(idx.shape[0]):
            translated.append(" ".join([self.__int2word[int(index)]
                                        for index in idx[b]]))

        return translated

    def __load_embedding(self):
        """

        :return:
        """
        word2int_file = pd.read_csv("./embedding/word2int.csv")
        word2int_file = word2int_file.sort_values(by=0, axis=1)
        words = word2int_file.columns
        indices = word2int_file.iloc[0, :]

        if self.load_embed:

            self.__word2int = {word: idx for idx, word in enumerate(words)}
            self.__int2word = {idx: word for idx, word in enumerate(words)}

            embed_file = pd.read_csv(self.embed_path)
            embed_file_keys = [self.__word2int[word] for word in embed_file["word"]]
            embed_file.insert(1, "key", embed_file_keys)
            embed_file = embed_file.sort_values(by="key")
            vectors = embed_file.iloc[:, 2:].values

            self.word2vecs = torch.stack([torch.Tensor(vectors[int(idx)])
                                          for idx in indices], dim=0).to(self.device)
            self.word2vecs.requires_grad = self.train_embed

            self.embed_length = self.word2vecs.shape[1]

        else:

            self.__word2int = {word: idx for idx, word in zip(indices, words)}
            self.__int2word = {idx: word for idx, word in zip(indices, words)}

            num_words = len(list(self.__word2int.keys()))
            self.word2vecs = torch.stack([torch.rand(self.embed_length)
                                          for _ in range(num_words)], dim=0)
            self.word2vecs.requires_grad = self.train_embed

    def __getitem__(self, item):
        index = self.__word2int[item]
        return self.word2vecs[index]

    @property
    def int2word(self):
        return self.__int2word

    @property
    def word2int(self):
        return self.__word2int

    @property
    def vector_dim(self):
        return self.embed_length
