from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ssal.training.trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Trainer for standard supervised image classification."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: Optimizer,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
        )

        self.optimizer = optimizer

        self.criterion = (
            criterion
            if criterion is not None
            else nn.CrossEntropyLoss()
        )

    def train_epoch(
        self,
        loader: DataLoader[Any],
    ) -> dict[str, float]:
        """Train the model for one epoch."""

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            loss = self.criterion(
                logits,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader[Any],
    ) -> dict[str, float]:
        """Evaluate the model."""

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)

            loss = self.criterion(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }
