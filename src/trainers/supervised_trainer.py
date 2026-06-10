# src/trainers/supervised_trainer.py

import torch


class SupervisedTrainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        correct = 0
        total = 0

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

        return 100.0 * correct / total
