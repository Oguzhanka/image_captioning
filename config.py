import torch


class Params:
    def __init__(self):
        pass

    def params(self):
        return self.__dict__


class DataParams(Params):
    def __init__(self):
        super(DataParams, self).__init__()

        self.model_name = "vggrnn"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_source = "flickr"

        if self.data_source == "default":
            self.image_path = "./dataset/images/"
            self.dataset_path = "./dataset"
            self.embed_path = "./embedding/limited_glove_vectors.csv"
            self.url_path = "./dataset/img_url.csv"

        elif self.data_source == "flickr":
            self.image_path = "./dataset/flickr_set/flickr_images"
            self.dataset_path = "./dataset"
            self.embed_path = "./dataset/limited_glove_vectors.csv"
            self.url_path = "./dataset/img_url.csv"

        self.load_embed = "fasttext"
        self.train_embed = False
        self.embed_length = 300
        self.num_spec_chars = 0

        self.num_epochs = 1
        self.batch_size = 128
        self.sequence_length = 16
        self.word_length = 1004
        self.input_size = (224, 224)
        self.num_layers = 1
        self.min_num_captions = 3

        self.train_length = []
        self.validation_length = []
        self.test_length = []

        self.rnn_flow = "RNN"
        self.word_flow = False

        self.hidden_size = self.word_length if self.rnn_flow == "RNN" else 64


class VggRNNParams(Params):
    def __init__(self):
        super(VggRNNParams, self).__init__()
        self.pretrained_cnn = True
        self.trainable_cnn = False

        self.num_layers = 1

        self.optimizer_type = "SGD"
        self.optimizer_params = {"lr": 0.01}

        self.criterion_type = "CE"
        self.criterion_params = {}


class VggLSTMParams(Params):
    def __init__(self):
        super(VggLSTMParams, self).__init__()
        self.pretrained_cnn = True
        self.trainable_cnn = False

        self.num_layers = 1

        self.optimizer_type = "ADAM"
        self.optimizer_params = {"lr": 0.01}

        self.criterion_type = "CE"
        self.criterion_params = {}


class InceptionRNNParams(Params):
    def __init__(self):
        super(InceptionRNNParams, self).__init__()
        self.pretrained_cnn = True
        self.trainable_cnn = False

        self.num_layers = 1

        self.optimizer_type = "ADAM"
        self.optimizer_params = {"lr": 0.01}

        self.criterion_type = "CE"
        self.criterion_params = {}


class InceptionLSTMParams(Params):
    def __init__(self):
        super(InceptionLSTMParams, self).__init__()
        self.pretrained_cnn = True
        self.trainable_cnn = False

        self.num_layers = 1

        self.optimizer_type = "SGD"
        self.optimizer_params = {"lr": 0.01}

        self.criterion_type = "CE"
        self.criterion_params = {}
