import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # we have 32 * 32 * 3

        # Block 1
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Block 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 16 x 16 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 16 x 16 x 64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Block 3
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 8 x 8 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 8 x 8 x 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 4 x 4 x 256
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 4 x 4 x 256
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Pooling layer
        self.maxpool = nn.MaxPool2d(2)

        # Dropout layer
        self.drop = nn.Dropout2d(p=0.25)

        # Flatten layer
        self.flat = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(4 * 4 * 256, 1028)
        self.fc2 = nn.Linear(1028, 10)

        # Batch normalization layer for the fully connected layer
        self.batch1d = nn.BatchNorm1d(1028)

        # ReLU Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input: 32 x 32 x 3
        x = self.block_1(x)  # Output: 32 x 32 x 32

        x = self.maxpool(x)  # Output: 16 x 16 x 32

        x = self.drop(x)

        x = self.block_2(x)  # Output: 16 x 16 x 64

        x = self.maxpool(x)  # Output: 8 x 8 x 64

        x = self.drop(x)

        x = self.block_3(x)  # Output: 8 x 8 x 128

        x = self.maxpool(x)  # Output: 4 x 4 x 128

        x = self.drop(x)

        x = self.block_4(x)  # Output: 4 x 4 x 256

        x = self.flat(x)  # Output: 1 x (4 * 4 * 256) = 1 x 4096

        x = self.fc1(x)  # Output: 1 x 1028

        x = self.batch1d(x)

        x = self.relu(x)

        x = self.fc2(x)  # Output: 1 x 10

        return x


def create_advanced_skip_connection_conv_cifar():
    return Model()
