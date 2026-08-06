import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def train(self, epochs=1):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for images, labels in self.train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

        return self.model.state_dict()