# src/main.py

import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from datasets.cifar10 import CIFAR10Dataset
from models.backbones.resnet18 import ResNet18Classifier
from trainers.supervised_trainer import SupervisedTrainer
from utils.seed import set_seed


def main():

    set_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = CIFAR10Dataset()

    train_dataset = dataset.get_train_dataset()
    test_dataset = dataset.get_test_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
    )

    model = ResNet18Classifier(
        num_classes=dataset.num_classes()
    )

    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    epochs = 5

    for epoch in range(epochs):

        loss = trainer.train_epoch(
            train_loader
        )

        accuracy = trainer.evaluate(
            test_loader
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()
