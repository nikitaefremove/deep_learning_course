import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader


# Normalization function. Return mean and standard deviation
def get_normalize(features: torch.Tensor):
    # size [N, C, H, W]
    # N is amount of objects, C — amount of chanel, H, W — size of image
    # return mean by chanel and standard deviation by chanel

    mean = features.mean(dim=(0, 2, 3))
    std = features.std(dim=(0, 2, 3))

    return mean, std


# Function for augmentation.
# Resize image,  make augmentation for train (crop, b&w, blur, random)
# Make tensor from train and test and normalize
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


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    predictions = []
    model = model.to(device)
    model.eval()  # set the model to evaluation mode

    with torch.no_grad():  # deactivate autograd
        for x, _ in loader:  # we do not need 'y' for predictions
            x = x.to(device)
            output = model(x)  # forward pass
            preds = output.argmax(dim=1)  # get the predicted classes
            predictions.append(preds)

    return torch.cat(predictions)
