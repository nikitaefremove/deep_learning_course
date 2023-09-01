#!g1.1


def conv_plus_conv(in_channels: int, out_channels: int):
    """
    Makes UNet block
    :param in_channels: input channels
    :param out_channels: output channels
    :return: UNet block
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
    )


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        base_channels = 32

        self.down1 = conv_plus_conv(3, base_channels)
        self.down2 = conv_plus_conv(base_channels, base_channels * 2)
        self.down3 = conv_plus_conv(base_channels * 2, base_channels * 4)

        self.up1 = conv_plus_conv(base_channels * 2, base_channels)
        self.up2 = conv_plus_conv(base_channels * 4, base_channels)
        self.up3 = conv_plus_conv(base_channels * 3, base_channels)

        self.bottleneck = conv_plus_conv(base_channels * 4, base_channels * 4)

        self.out = nn.Conv2d(in_channels=base_channels, out_channels=3, kernel_size=1)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x.shape = (N, N, 3)

        residual1 = self.down1(x)  # N, N, base_channels
        x = self.downsample(residual1)  # N // 2, N // 2, base_channels

        residual2 = self.down2(x)  # N // 2, N // 2, base_channels * 2
        x = self.downsample(residual2)  # N // 4, N // 4, base_channels * 2

        residual3 = self.down3(x)  # N // 4, N // 4, base_channels * 4
        x = self.downsample(residual3)  # N // 8, N // 8, base_channels * 4

        # LATENT SPACE DIMENSION DIM = N // 8
        x = self.bottleneck(x)  # N // 8, N // 8, base_channels * 4
        # LATENT SPACE DIMENSION DIM = N // 8

        x = nn.functional.interpolate(x, scale_factor=2)  # N // 4, N // 4, base_channels * 4
        x = torch.cat((x, residual3), dim=1)  # N // 4, N // 4, base_channels * 8
        x = self.up3(x)  # N // 4, N // 4, base_channels

        x = nn.functional.interpolate(x, scale_factor=2)  # N // 2, N // 2, base_channels
        x = torch.cat((x, residual2), dim=1)  # N // 2, N // 2, base_channels * 4
        x = self.up2(x)  # N // 2, N // 2, base_channels

        x = nn.functional.interpolate(x, scale_factor=2)  # N, N, base_channels
        x = torch.cat((x, residual1), dim=1)  # N, N, base_channels * 2
        x = self.up1(x)  # N, N, base_channels

        x = self.out(x)  # N, N, 3  # Change here

        return x
