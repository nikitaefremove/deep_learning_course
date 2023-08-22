import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        total_loss += loss.item()
        loss.backward()
        print(f'MSE на шаге {i + 1} {loss.item():.5f}')
        optimizer.step()

    return total_loss / len(data_loader)
