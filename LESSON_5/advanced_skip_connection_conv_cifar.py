import torch.nn as nn


class create_simple_conv_cifar(nn.Module):
    def __init__(self):
        super(create_simple_conv_cifar, self).__init__()

        # we have 32 * 32 * 3

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 16 x 16 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 16 x 16 x 64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 8 x 8 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 8 x 8 x 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(2)

        self.drop = nn.Dropout2d(p=0.25)

        self.flat = nn.Flatten()

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(4 * 4 * 128, 512)

        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        return self.net(x)
