import torch
import torch.nn as nn
from losses import loss_dict
from optimizers import optimizer_dict
from embedding.embedding import Embedding


class BaseModel(nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.params = params
        self.criterion_type = params["criterion_type"]
        self.class_weights = torch.Tensor(self.params["class_weights"]).to(self.params["device"])
        self.class_weights.requires_grad = False
        self.class_weights[-1] = 0.0
        self.class_weights[-3] = 0.0

        self.image_process = None
        self.word_process = None
        self.embedding = Embedding(**params)

        self._construct_model()
        self.criterion = None
        self.optimizer = optimizer_dict[params["optimizer_type"]](self.parameters(),
                                                                  **params["optimizer_params"])

    def forward(self, input_, batch_y=None):
        """

        :param input_: (b, d, m, n)
        :param batch_y: (b, l)
        :return:
        """
        image_features = self.image_process(input_)             # (b, f)
        generated_words = self.word_process(image_features)     # (b, l, w)
        return generated_words

    def fit(self, batch_x, batch_y):
        """
        Single step prediction.

        :param batch_x: (b, d, m, n) Input image.
        :param batch_y: (b, l) Caption of the input image.
        :return:
        """
        batch_x = batch_x.to(self.params["device"])
        batch_y = batch_y.to(self.params["device"])

        generated_words = self(batch_x, batch_y)
        self.optimizer.zero_grad()

        loss = 0
        self.criterion = loss_dict[self.criterion_type](weight=self.class_weights)

        for l in range(self.params["sequence_length"]):
            cur_word = generated_words.narrow(1, l, 1).squeeze(dim=1)
            loss += self.criterion(cur_word, batch_y[:, l]) / batch_x.shape[0]
        loss.backward(retain_graph=True)

        self.optimizer.step()
        return loss.item()

    def caption(self, batch_x):
        """

        :param batch_x: (b, d, m, n)
        :return:
        """
        batch_x = batch_x.to(self.params["device"])
        with torch.no_grad():
            image_features = self.image_process(batch_x)                 # (b, f)
            generated_words = self.word_process.caption(image_features)  # (b, l)
            caption = self.embedding.translate(generated_words)
        return caption
