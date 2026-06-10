from abc import ABC, abstractmethod


class BaseDataset(ABC):

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def num_classes(self):
        pass