import torch


def function01(tensor: torch.Tensor, count_over: str) -> torch.Tensor:
    if count_over == 'columns':
        # mean value of columns
        return tensor.mean(dim=0)
    if count_over == 'rows':
        # mean value of rows
        return tensor.mean(dim=1)
    else:
        raise ValueError(f"Invalid value for count_over: {count_over}. It should be 'columns' or 'rows'.")
