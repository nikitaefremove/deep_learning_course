import torch


def get_normalize(features: torch.Tensor):
    # размер [N, C, H, W]
    # N это количество объектов, C — количество каналов, H, W — размеры изображений
    # будет возвращать поканальное среднее и поканальное стандартное отклонение

    mean = features.mean(dim=(0, 2, 3))
    std = features.std(dim=(0, 2, 3))

    return mean, std
