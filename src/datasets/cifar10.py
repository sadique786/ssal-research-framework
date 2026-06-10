from torchvision import datasets, transforms

from .base_dataset import BaseDataset


class CIFAR10Dataset(BaseDataset):

    def __init__(self, root="./data"):

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.root = root

    def get_train_dataset(self):

        return datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.train_transform,
        )

    def get_test_dataset(self):

        return datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.test_transform,
        )

    def num_classes(self):
        return 10