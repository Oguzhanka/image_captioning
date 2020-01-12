import torch
import torch.nn as nn
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


class VggAtt(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        super(VggAtt, self).__init__(kwargs["input_size"],
                                     kwargs.get("trainable_cnn", True),
                                     device=kwargs["device"])
        self.rnn_flow = kwargs["rnn_flow"]
        self.model = VggAttention(pretrained=kwargs.get("pretrained_cnn", False),
                                  device=kwargs["device"])
        self._set_classifier(output_dim)


class VggAttention(nn.Module):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(VggAttention, self).__init__()
        layer_params = [64, 64, 128, 128, 256,
                        256, 256, 'M', 512, 512,
                        512, 'M', 512, 512, 512,
                        'M', 512, 'M', 512, 'M']
        self.features = self._make_layers(layer_params)
        self.classifier = nn.Linear(512, 10).to(kwargs["device"])

        self.l1 = nn.Sequential(*list(self.features)[:22]).to(kwargs["device"])
        self.l2 = nn.Sequential(*list(self.features)[22:32]).to(kwargs["device"])
        self.l3 = nn.Sequential(*list(self.features)[32:42]).to(kwargs["device"])

        self.u1 = nn.Conv2d(256, 1, 1).to(kwargs["device"])
        self.u2 = nn.Conv2d(512, 1, 1).to(kwargs["device"])
        self.u3 = nn.Conv2d(512, 1, 1).to(kwargs["device"])

        self.conv_out = nn.Sequential(*list(self.features)[42:50]).to(kwargs["device"])
        self.fc1 = nn.Linear(18432, 512).to(kwargs["device"])

        self.fc1_l1 = nn.Linear(512, 256).to(kwargs["device"])
        self.fc1_l2 = nn.Linear(512, 512).to(kwargs["device"])
        self.fc1_l3 = nn.Linear(512, 512).to(kwargs["device"])

        self.fc2 = nn.Linear(256 + 512 + 512, 10).to(kwargs["device"])

        self.sf_1 = nn.Softmax(dim=2).to(kwargs["device"])
        self.sf_2 = nn.Softmax(dim=2).to(kwargs["device"])

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)

        conv_out = self.conv_out(l3)
        fc1 = self.fc1(conv_out.view(conv_out.size(0), -1))
        fc1_l1 = self.fc1_l1(fc1)
        fc1_l2 = self.fc1_l2(fc1)
        fc1_l3 = self.fc1_l3(fc1)

        att1 = self._compatibility_fn(l1, fc1_l1, level=1)
        att2 = self._compatibility_fn(l2, fc1_l2, level=2)
        att3 = self._compatibility_fn(l3, fc1_l3, level=3)

        g1 = self._weighted_combine(l1, att1)
        g2 = self._weighted_combine(l2, att2)
        g3 = self._weighted_combine(l3, att3)

        g = torch.cat((g1, g2, g3), dim=1)
        out = self.fc2(g)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers).to(self.kwargs["device"])

    def _compatibility_fn(self, l, g, level):
        att = l + g.unsqueeze(2).unsqueeze(3)

        if level == 1:
            u = self.u1
        elif level == 2:
            u = self.u2
        elif level == 3:
            u = self.u3
        att = u(att)

        size = att.size()
        att = att.view(att.size(0), att.size(1), -1)
        att = self.sf_2(att)
        att = att.view(size)

        return att

    def _weighted_combine(self, l, att_map):
        g = l * att_map
        return g.view(g.size(0), g.size(1), -1).sum(2)

