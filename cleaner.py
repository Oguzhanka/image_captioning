from PIL import Image
import pandas as pd
import numpy as np
import string
import os


def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(str.maketrans("", "", string.punctuation))
    return text_no_punctuation


def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1 or word == "A" or word == "a":
            text_len_more_than1 += " " + word.lower()
    return text_len_more_than1


def remove_numeric(text, printTF=False):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if printTF:
            print("    {:10} : {:}".format(word, isalpha))
        if isalpha:
            text_no_numeric += " " + word
    return text_no_numeric


filename = "./dataset/flickr/Flickr8k.lemma.token.txt"
file = open(filename, "rb")
image_id = []
image_name = []
caption = []
sep_caption = []

words = np.array([])

for line in file:
    line = str(line)[2:]
    image_id.append(int(line.split("_")[0]))
    image_name.append(line.split("#")[0])
    cap = line.split("\\t")[1][:-2].replace("\\", "").replace("\\\\", "")

    cap = remove_punctuation(cap)
    cap = remove_single_character(cap)
    cap = remove_numeric(cap).split(" ")

    cap.append("x_END_")
    cap.extend(["x_NULL_" for _ in range(16-len(cap))])

    tiled_cap = ["x_START_"]
    tiled_cap.extend(cap)
    tiled_cap = " ".join(tiled_cap)
    caption.append(tiled_cap.split(" "))

    unique_words = np.unique(tiled_cap.split(" "))
    words = np.concatenate([words, unique_words], axis=0)
    sep_caption.append(tiled_cap.split(" "))

caps = np.array(caption)
sep_caps = np.array(sep_caption)
words = np.unique(words)
word2int = {word: idx for idx, word in enumerate(words)}
key_frame = pd.DataFrame(data=[list(range(len(words)))], columns=words)
key_frame.to_csv("./dataset/flickr/word2int.csv", index=False)


int_captions = np.zeros_like(sep_caps)
for un_word in unique_words:
    int_captions[sep_caps == un_word] = word2int[un_word]
col_names = []
col_names.extend(["word" + str(i) for i in range(16)])
data_frame = pd.DataFrame([int_captions[:, i] for i in range(16)], columns=col_names)
data_frame.to_csv("./dataset/flickr/captions.csv", index=False)

id_frame = pd.DataFrame(image_id, columns=["im_addr"])
id_frame.to_csv("./dataset/flickr/imid.csv", index=False)

for im_id, im_name in zip(image_id, image_name):
    im = Image.open("./dataset/flickr/images/" + im_name)
    os.remove("./dataset/flickr/images/" + im_name)
    im.save("./dataset/flickr/images/" + str(im_id))

