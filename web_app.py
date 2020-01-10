import random
import config
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
import matplotlib.pyplot as plt
from data_extractor import get_data
from models.vgg_rnn import VggRNN, VggLSTM
from models.inception_rnn import InceptionRNN, InceptionLSTM
from batch_generator import BatchGenerator


model = None
batch_gen = None

models = {"vggrnn": {"model": VggRNN,
                     "params": config.VggRNNParams},

          "vgglstm": {"model": VggLSTM,
                      "params": config.VggLSTMParams},

          "inceptionrnn": {"model": InceptionRNN,
                           "params": config.InceptionRNNParams},

          "inceptionlstm": {"model": InceptionLSTM,
                            "params": config.InceptionLSTMParams}
          }


app = Flask(__name__)


@app.route('/train', methods=['GET', 'POST'])
def train():
    global model
    global batch_gen

    epochs = request.args.get('epochs', type=int)

    for e in range(epochs):
        print("Epoch num: " + str(e))
        for idx, (im, cap) in enumerate(batch_gen.generate('train')):
            if idx == 100:
                break
            loss = model.fit(im, cap)
            print("\rTraining: " + str(loss) + " [" + "="*idx, end="", flush=True)
        print("]")

    return "Done!\n"


@app.route("/caption", methods=['GET', 'POST'])
def caption():
    num_caps = request.args.get('caps', type=int)

    (ims, caps) = next(batch_gen.generate("train"))
    generated_captions = model.caption(ims)
    random_idx = random.sample(list(range(parameters["batch_size"])), num_caps)

    for idx in random_idx:
        cap = generated_captions[idx]
        img = ims[idx]
        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.tight_layout()
        plt.title(cap)
        plt.show()

    return "Done!\n"


@app.route("/reset")
def reset():
    global model
    global batch_gen

    model = models[parameters["model_name"]]["model"](parameters)
    batch_gen = BatchGenerator(**parameters)

    return "Reset!\n"


if __name__ == '__main__':

    data_parameters = config.DataParams().__dict__
    model_parameters = models[data_parameters["model_name"]]["params"]().__dict__
    parameters = model_parameters.copy()
    parameters.update(data_parameters)

    get_data(parameters)

    if parameters["data_source"] == "default":
        captions = pd.read_csv("./dataset/captions.csv")

        captions = np.array(captions)
        histogram = [(captions == i).sum() for i in range(1004)]
        histogram = np.array(histogram)
        histogram[histogram > 50000] = 50000
        smooth_histogram = np.log(histogram)
        inverted_weights = 1 / smooth_histogram
        scaled = inverted_weights + (inverted_weights - 0.13) * 12 + 0.4
    else:
        scaled = None
    parameters.update({"weight": scaled})

    model = models[parameters["model_name"]]["model"](parameters)
    batch_gen = BatchGenerator(**parameters)

    app.run(host='0.0.0.0')
