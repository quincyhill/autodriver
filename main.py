import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

BATCH_SIZE = 64

def download_the_datasets():
    # Download training data from open datasets
    training_data = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    # Download test data from open datasets
    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    

def get_the_datasets():
    training_data = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=False,
        transform=ToTensor()
    )
    
    # Download test data from open datasets
    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    
    return training_data, test_data
    
    
def print_shape(data_set):
    
    dataloader = DataLoader(data_set, batch_size=BATCH_SIZE)

    for X, y in dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y [N]: {y.shape}")
        break

def iterate_and_visualize_dataset(data_set):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }   

    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data_set), size=(1,)).item()
        img, label = data_set[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

training_data, test_data = get_the_datasets()


iterate_and_visualize_dataset(training_data)
