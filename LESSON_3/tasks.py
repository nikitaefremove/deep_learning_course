import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from matplotlib import cm
from time import perf_counter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Optimizer


# Get count of parameters of convolutional Neural Network
def count_parameters_conv(in_channels: int,
                          out_channels: int,
                          kernel_size: int,
                          bias: bool):
    if bias == True:
        return (in_channels * kernel_size ** 2 + 1) * out_channels

    else:
        return (in_channels * kernel_size ** 2) * out_channels


# Train model. Return average of loss
def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        total_loss += loss.item()
        loss.backward()
        print(f'{loss.item():.5f}')
        optimizer.step()

    return total_loss / len(data_loader)


# Evaluate function. Return average of loss
@torch.inference_mode()
def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    model.eval()
    total_loss = 0

    for i, (x, y) in enumerate(data_loader):
        output = model(x)
        loss = loss_fn(output, y)
        total_loss += loss.item()

    return total_loss / len(data_loader)


# Load MNIST Dataset. Split by train and validation parts.
mnist_train = MNIST(
    "../datasets/mnist",
    train=True,
    download=True,
    transform=T.ToTensor()
)

mnist_valid = MNIST(
    "../datasets/mnist",
    train=False,
    download=True,
    transform=T.ToTensor()
)

# Load MNIST datasets with DataLoader.
# Add batches, and shuffle data for better result

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
valid_loader = DataLoader(mnist_valid, batch_size=64, shuffle=False)


# Create model function
def create_mlp_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(4 * 4 * 64, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    return model


model = create_mlp_model()
