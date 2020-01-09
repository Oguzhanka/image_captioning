from torchvision.models import vgg11, vgg16, vgg19
from image_modules.base_image_module import BaseImageModule


class VGG11(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        super(VGG11, self).__init__(kwargs["input_size"],
                                    kwargs.get("trainable_cnn", True),
                                    device=kwargs["device"])
        self.rnn_flow = kwargs["rnn_flow"]
        self.model = vgg11(pretrained=kwargs.get("pretrained_cnn", False)).to(kwargs["device"])
        self._set_classifier(output_dim)


class VGG16(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        super(VGG16, self).__init__(kwargs["input_size"],
                                    kwargs.get("trainable_cnn", True),
                                    device=kwargs["device"])
        self.rnn_flow = kwargs["rnn_flow"]
        self.model = vgg16(pretrained=kwargs.get("pretrained_cnn", False)).to(kwargs["device"])
        self._set_classifier(output_dim)


class VGG19(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        super(VGG19, self).__init__(kwargs["input_size"],
                                    kwargs.get("trainable_cnn", True),
                                    device=kwargs["device"])
        self.rnn_flow = kwargs["rnn_flow"]
        self.model = vgg19(pretrained=kwargs.get("pretrained_cnn", False)).to(kwargs["device"])
        self._set_classifier(output_dim)
