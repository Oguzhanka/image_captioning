import random

from load_data import LoadData
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from embedding.embedding import Embedding


class BatchGenerator(LoadData):

    def __init__(self, dataset_path, images_path, **kwargs):
        super(BatchGenerator, self).__init__(dataset_path, images_path)

        self.batch_size = kwargs.get('batch_size', 64)
        self.shuffle = kwargs.get('shuffle', True)
        self.num_works = kwargs.get('num_works', 4)
        self.test_ratio = kwargs.get('test_ratio', 0.1)
        self.val_ratio = kwargs.get('val_ratio', 0.1)
        self.embed_path = kwargs.get('embedding_path', 'embedding')
        self.vector_dim = kwargs.get('vector_dim', 300)
        self.use_transform = kwargs.get('use_transform', True)

        self.embedding = Embedding(self.embed_path, self.vector_dim)

        self.dataset_dict = {i: self.__create_data(batch_format=i)[0]
                             for i in ['embedding', 'integer', 'word']}

        self.dataloader_dict = {i: self.__create_data(batch_format=i)[1]
                                for i in ['embedding', 'integer', 'word']}

    def generate(self, data_type, batch_format='embedding'):
        """
        :param data_type: can be 'test', 'train' and 'validation'
        :param batch_format: can be 'integer', 'embedding'
        :return: img tensor, label numpy_array
        """
        selected_loader = self.dataloader_dict[batch_format][data_type]
        yield from selected_loader

    def __create_data(self, batch_format):
        data_dict = self.__split_data()

        if self.use_transform:
            im_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            im_transform = None

        im_datasets = {}
        for i in ['test', 'train', 'validation']:
            params = {
                'image_path_names': data_dict[i],
                'captions_int': self.captions_int,
                'captions_word': self.caption_words,
                'im_addr': self.image_addr,
                'embedding': self.embedding,
                'batch_format': batch_format,
                'transformer': im_transform
            }
            im_datasets[i] = ImageDataset(params)

        im_dataloaders = {i: DataLoader(im_datasets[i],
                                        batch_size=self.batch_size,
                                        shuffle=self.shuffle,
                                        num_workers=self.num_works)
                          for i in ['test', 'train', 'validation']}

        return im_datasets, im_dataloaders

    def __split_data(self):
        dataset_length = len(self.image_paths)

        random.shuffle(self.image_paths)

        test_count = int(dataset_length * self.test_ratio)
        val_count = int(dataset_length * self.val_ratio)

        data_dict = dict()
        data_dict['test'] = self.image_paths[:test_count]
        data_dict['validation'] = self.image_paths[test_count:test_count + val_count]
        data_dict['train'] = self.image_paths[test_count + val_count:]

        return data_dict