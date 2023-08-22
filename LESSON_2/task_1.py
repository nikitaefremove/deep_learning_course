import torch.nn as nn


def create_model():
    model = nn.Sequential(
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 1))

    return model
