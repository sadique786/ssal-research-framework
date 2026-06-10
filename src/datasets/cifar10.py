# src/datasets/cifar10.py

from torchvision import datasets
from torchvision import transforms

from .base_dataset import BaseDataset


class CIFAR10Dataset(BaseDataset):
    """
    CIFAR10 dataset wrapper.
    """

    def __init__(self, root: str = "./data"):

        self.root = root

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    def get_train_dataset(self):
        return datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform_train,
        )

    def get_test_dataset(self):
        return datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform_test,
        )

    def num_classes(self):
        return 10
