import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data_loaders(data_dir="chest_xray"):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=transform
    )

    # LIMIT DATASET SIZE FOR FAST TRAINING
    train_dataset = Subset(train_dataset, range(100))
    test_dataset = Subset(test_dataset, range(50))

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    return train_loader, test_loader