import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def create_model():
    model = nn.Sequential(
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 1))

    return model


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
