import torch
import torchvision.transforms as T


def get_normalize(features: torch.Tensor):
    # size [N, C, H, W]
    # N is amount of objects, C — amount of chanel, H, W — size of image
    # return mean by chanel and standard deviation by chanel

    mean = features.mean(dim=(0, 2, 3))
    std = features.std(dim=(0, 2, 3))

    return mean, std


def get_augmentations(train: bool = True) -> T.Compose:
    means = (0.49139968, 0.48215841, 0.44653091)
    stds = (0.24703223, 0.24348513, 0.26158784)
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.CenterCrop(10),
            T.Grayscale(),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.RandAugment(),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
        )
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
        )
