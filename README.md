# image_captioning

Image captioning project.
    
## Table of contents
* [Requirements](#Requirements)
* [General info](#General info)
* [Usage](#Usage)
* [Single run](#Single run)
* [Model architecture](#Model architecture)
* [Configuration file](#Configuration file)
* [Input data format](#Input data format)
    
## Requirements
   
This repository was prepared using Python3.6. Below packages are required to run the code. It is recommended to use the
specified versions of the packages.

* pandas==0.23.4
* torchvision==0.4.0
* matplotlib==3.0.3
* torch==1.2.0
* pytorch_nlp==0.5.0
* numpy==1.17.4
* Flask==1.1.1
* Pillow==7.0.0


## General info

This repository contains a configurable image captioning model that is based on Convolutional Neural Networks and
Recurrent Neural Networks. The code allows the user to change the model architecture and tune hyper-parameters.
Configurable parameters are listed with their descriptions in the Configuration section. The user can run the training
session which will fit the model on the data and save it on a file. Alternatively, a web application script (web_app.py)
initializes a model and waits for user input to either fit the model or caption an image. The user can check the model
performance at every stage.


## Installation

First of all, Python3.6+ is required to use this repository. It can be installed with the following command.

```
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update
$ sudo apt install python3.6
```

After the Python installation, required packages should be installed with the following command.

```
$ sudo pip3 install -r requirements.txt
```

## Usage

As mentioned in the previous section, two modes are available.


#### Single run
This mode fits the model with the specified configurations on the dataset. After the fitting is completed, test score
will be computed and the model will be saved to "./model_objects/" directory. This mode can be used with the following 
command:

```
$ python3 run.py
```

#### Running with the web application

This mode initializes a model and stores in memory. The user can fit the model for several epochs and test the model at
any time. In single run mode, the model might overfit on the data. However, in this mode, it is possible to asses the
performance at every stage. This model can be used with the following command:

```
$ python3 web_app.py
```

This script will expect user requests to perform some operations. These operations are described below.


* /train    This request will trigger the model fitting flow. Accepts number of epochs as argument. An example command
is shown below.

```
$ curl http://127.0.0.1/train?epochs=5
```

* /caption  This request will select a set of random images from the provided dataset and captions them with the current
state of the model. Accepts the number of images as argument. An example command is shown below.

```
$ curl http://127.0.0.1/caption?caps=5
```

* /reset    This command re-initializes a model object.


## Model architecture

* Overall architecture can be summarized as a CNN followed by an RNN model. CNN extracts the visual features and RNN
sequentially generates words to form captions.
 
* CNN model can be a pretrained VGG, Inception or ResNet model. Although it is possible to train CNN from scratch, it
would require too much time and memory. Code structure allows adding a custom CNN model extending the base image module.
 
* RNN model can be a multilayer standard RNN or a multilayer LSTM. Code structure allows adding a custom recurrent layer
extending the base word processing module.

* Word embeddings are implemented in "embedding.py". Embedding vectors are either generated as random vectors and fitted
on data or initialized as pretrained embedding models.

* Embedding vectors can be Fasttext vectors, Glove vectors or random vectors.

* A dense module converts the model state to word vectors. This dense module is implemented in dense_1.py and dense_2.py
which can be modified as long as the input and output dimensions match.

* Two main data flows are implemented for word processing. In the first approach, visual features extracted by the CNN
are fed to an RNN's hidden state vector. Starting from the x_START_ token, fixed length captions are generated. 
(rnn_flow = "RNN"). In the second approach, visual features are fed to a dense layer along with the RNNs hidden state 
(rnn_flow = "parallel").

* Model can be trained with teacher-forcing method or with multi-step prediction mode.


## Configuration file

TODO


## Input data format
