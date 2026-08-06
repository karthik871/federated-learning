import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_pneumonia import PneumoniaModel
from data.pneumonia_loader import get_data_loaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, test_loader = get_data_loaders("chest_xray")

# Load model
model = PneumoniaModel()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

print("Training finished")