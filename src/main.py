import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.datasets.cifar10 import CIFAR10Dataset

from src.models.backbones.resnet18 import ResNet18Classifier

from src.trainers.supervised_trainer import SupervisedTrainer

from src.utils.seed import set_seed


def main():

    set_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

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
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    trainer.fit(epochs=5)


if __name__ == "__main__":
    main()