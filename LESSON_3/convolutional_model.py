import torch
import torch.nn as nn
import torchvision.transforms as T
from time import perf_counter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Optimizer