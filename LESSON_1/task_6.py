import torch


def function02(dataset: torch.Tensor):
    n_features = dataset.shape[1]
    return torch.rand(n_features, requires_grad=True, dtype=torch.float32)
