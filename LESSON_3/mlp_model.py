import torch
import torch.nn as nn
import torchvision.transforms as T
from time import perf_counter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Optimizer


def create_mlp_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),  # 28*28 - размер изображений MNIST
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)  # 10 классов в MNIST (цифры от 0 до 9)
    )

    return model

