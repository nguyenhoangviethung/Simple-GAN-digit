import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

batch_size = 128


transform = transforms.Compose([
    transforms.ToTensor,
    transforms.Normalize([0.5], [0.5])
])

def load_MNIST_data(root='./data', transform=transform):
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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    print(load_MNIST_data())