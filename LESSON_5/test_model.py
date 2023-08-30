import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # we have 32 * 32 * 3 after resizing

        # BLOCK 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 32 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # BLOCK 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # 16 x 16 x 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25)
        )

        # BLOCK 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 16 x 16 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # BLOCK 4
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 8 x 8 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25)
        )

        # BLOCK 5
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 8 x 8 x 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # BLOCK 6
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 4 x 4 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 128, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Blocks
        x = self.block1(x)  # 32 x 32 x 32
        x = self.block2(x) + x  # 16 x 16 x 32
        x = self.block3(x)  # 16 x 16 x 64
        x = self.block4(x) + x  # 8 x 8 x 64
        x = self.block5(x)  # 8 x 8 x 128
        x = self.block6(x) + x  # 4 x 4 x 128

        # Fully connected layers
        x = self.fc(x)  # 4 x 4 x 128 -> 10

        return x


def create_advanced_skip_connection_conv_cifar():
    return Model()
