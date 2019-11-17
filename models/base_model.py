import torch.nn as nn
import torch.optim as opt


class BaseModel(nn.Module):
    def __init__(self, data_params, params):
        super(BaseModel, self).__init__()
        self.params = params
        self.input_size = data_params["input_size"]
        self.batch_size = data_params["batch_size"]*data_params["min_num_captions"]
        self.sequence_length = data_params["sequence_length"]
        self.modules = None

        self.optimizer = None
        self.criterion = None

    def fit(self, x, y):
        y_hat = self(x)

        def closure():
            self.optimizer.zero_grad()
            loss = 0.0
            for s in range(self.sequence_length):
                word_pred = y.narrow(1, s, 1).squeeze()
                word_hat = y_hat.narrow(1, s, 1).squeeze()
                loss += self.criterion(word_hat, word_pred)
            loss.backward()
            print(loss.item())
            return loss

        self.optimizer.step(closure)

    def update_optimizer(self, new_params):
        for key, value in new_params.items():
            self.optimizer.__setattr__(key, value)

    def construct_optimizer(self):
        opt_type = self.params["optimizer_type"]
        crit_type = self.params["criterion_type"]

        optimizer_dict = {"sgd": opt.SGD,
                          "adam": opt.Adam,
                          "lr_sch": opt.lr_scheduler}

        criterion_dict = {"mse": nn.MSELoss,
                          "cross_ent": nn.CrossEntropyLoss}

        self.optimizer =\
            optimizer_dict[opt_type](**self.params["optimizer_params"][opt_type],
                                     params=self.parameters())
        self.criterion =\
            criterion_dict[crit_type](**self.params["criterion_params"][crit_type])
