import torch
import torch.nn as nn
import torchvision.transforms as T
from time import perf_counter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Optimizer


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
