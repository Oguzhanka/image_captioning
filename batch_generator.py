import os
import torch
import random
import skimage
import numpy as np
import pandas as pd
from skimage import io


class BatchGenerator:

    def __init__(self, data_path, batch_size, min_num_captions, im_size, sequence_length, word_length):
        self.data_path = data_path
        self.batch_size = batch_size
        self.min_num_captions = min_num_captions
        self.im_size = im_size

        self.sequence_length = sequence_length
        self.word_length = word_length

        self.image_names = os.listdir(self.data_path["image_path"])
        self.caption_ids = pd.read_csv(self.data_path["caption_path"])
        self.caption_sentence = pd.read_csv(self.data_path["sentence_path"])

    def __next__(self):
        idx = self.__get_sample_indices()
        images, captions = self.__read_data_fs(idx)

        images = torch.Tensor(images)
        images = images.permute(dims=(0, 3, 1, 2))

        # captions = np.apply_along_axis(self.__batch_one_hot, axis=1, arr=captions)

        captions = torch.Tensor(captions).type(torch.LongTensor)
        return images, captions

    def __read_data_fs(self, idx):
        sampled_image_names = [image_name for image_name in self.image_names
                               if image_name.split("_")[0] in idx]

        while len(sampled_image_names) != self.batch_size:
            new_sample_idx = str(random.randint(0, len(self.image_names)))

            for file_name in self.image_names:
                if file_name.split("_")[0] == new_sample_idx:
                    sampled_image_names.append(file_name)
                    break

        # image_orders = [sampled_image.split("_")[0]
        #                 for sampled_image in sampled_image_names]
        image_ids = [int(sampled_image.split("_")[1].split(".")[0])
                     for sampled_image in sampled_image_names]

        sampled_images = [io.imread(self.data_path["image_path"] + sampled_image_name)
                          for sampled_image_name in sampled_image_names]
        rescaled_images = [skimage.transform.resize(image, self.im_size) for image in sampled_images]
        expanded_images = [np.expand_dims(image, axis=2) if len(image.shape) != 3 else image
                           for image in rescaled_images]
        gray_scaled_images = [image if image.shape[2] == 3 else np.repeat(image, 3, 2)
                              for image in expanded_images]
        image_batch = np.stack(gray_scaled_images, axis=0)
        batch_images = np.repeat(image_batch, repeats=self.min_num_captions,
                                 axis=0)

        caption_ids = [list(self.caption_ids.index[self.caption_ids["im_addr"] == image_id])
                       for image_id in image_ids]
        picked_caption_ids = [random.sample(caption_id, self.min_num_captions)
                              for caption_id in caption_ids]
        batch_captions = [np.array(self.caption_sentence.iloc[cap_ids, 1:])
                          for cap_ids in picked_caption_ids]
        batch_captions = np.concatenate(batch_captions, axis=0)

        return batch_images, batch_captions

    def __get_sample_indices(self):
        return [str(random.randint(0, len(self.image_names)))
                for _ in range(self.batch_size)]

    def __batch_one_hot(self, caption):

        one_hot_caption = []
        for word in caption:
            one_hot_word = np.zeros((1, self.word_length))
            one_hot_word[0, word-1] = 1
            one_hot_caption.append(one_hot_word)
        one_hot_caption = np.concatenate(one_hot_caption, axis=0)

        return one_hot_caption
