import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_MNIST_data(root='./data', batch_size=128, transform=transform):
    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader


def real_labels(size, device, smooth=True):
    return torch.full((size, 1), 0.9 if smooth else 1.0,
                      dtype=torch.float, device=device)

def fake_labels(size, device):
    return torch.zeros(size, 1, device=device)

if __name__ == "__main__":
    train_loader, test_loader = load_MNIST_data(batch_size=64)
    print(len(train_loader), len(test_loader))
