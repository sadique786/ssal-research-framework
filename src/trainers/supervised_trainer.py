import torch


class SupervisedTrainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
    ):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

    def train_epoch(self):

        self.model.train()

        running_loss = 0.0

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def evaluate(self):

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

        return 100.0 * correct / total

    def fit(self, epochs):

        for epoch in range(epochs):

            loss = self.train_epoch()

            accuracy = self.evaluate()

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {loss:.4f} "
                f"Accuracy: {accuracy:.2f}%"
            )