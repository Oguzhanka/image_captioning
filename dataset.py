import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):

    def __init__(self, params):

        self.image_path_names = params['image_path_names']
        self.captions_int = params['captions_int']
        self.captions_word = params['captions_word']
        self.im_addr = params['im_addr']
        self.transformer = params['transformer']

    def __len__(self):
        return len(self.image_path_names)
    
    def __getitem__(self, idx):
        
        im_path = self.image_path_names[idx]
        image_name = im_path.split('/')[-1].split("_")[0]
        image_id = int(image_name.split('.')[0])

        caption_idx = self.im_addr[self.im_addr['im_addr'] == image_id].index
        selected_caption_idx = np.random.choice(caption_idx.values)

        target_captions = self.captions_int.iloc[selected_caption_idx].values[1:]

        image = Image.open(im_path)
        image = image.convert('RGB')
        if self.transformer is not None:
            image = self.transformer(image)

        return image, target_captions
