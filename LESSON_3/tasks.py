import torch
import torch.nn as nn
import torchvision.transforms as T
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
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),  # 28*28 - размер изображений MNIST
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)  # 10 классов в MNIST (цифры от 0 до 9)
    )

    return model


def create_conv_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(4 * 4 * 64, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )


model = create_conv_model()

# Define a loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

num_epoch = 15
train_loss_history, valid_loss_history = [], []
valid_accuracy_history = []

start = perf_counter()

for epoch in range(num_epoch):
    train_loss = train(model, train_loader, optimizer, loss_fn)
    valid_loss = evaluate(model, valid_loader, loss_fn)

    # Compute validation accuracy
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in valid_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    valid_accuracy = correct / total

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)
    valid_accuracy_history.append(valid_accuracy)

    print(f"Epoch: {epoch + 1}, "
          f"Train Loss: {train_loss:.5f}, "
          f"Valid Loss: {valid_loss:.5f}, "
          f"Valid Accuracy: {valid_accuracy:.5f}")

# Save the weights of model
# torch.save(model.state_dict(), "mnist_mlp_model.pth")
torch.save(model.state_dict(), "mnist_cnn_model.pth")

# Load weights of model
# model.load_state_dict(torch.load("mnist_mlp_model.pth"))
