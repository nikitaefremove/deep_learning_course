import torch
from torch import nn


def function04(x: torch.Tensor, y: torch.Tensor):
    # Fix random seed
    torch.manual_seed(0)

    # Determine n_features, n_objects, weights
    n_features = x.shape[1]

    # Create a fully connected layer
    layer = nn.Linear(in_features=n_features, out_features=1, bias=False)

    # Ensure the layer uses the desired initial weights
    with torch.no_grad():
        layer.weight.copy_(torch.rand(n_features, requires_grad=True, dtype=torch.float32))

    # Define step size, max MSE, and max iterations
    step_size = 1e-2
    max_mse = 0.3
    max_iterations = 5000

    # Initial mse value (to enter the loop)
    mse = float('inf')
    iteration = 0

    # Loop for determine mse by gradient
    while mse >= max_mse:
        y_pred = layer(x).squeeze()  # Use the layer for predictions
        mse = torch.mean((y_pred - y) ** 2)

        print(f'Iteration {iteration}, MSE {mse.item():.5f}')
        mse.backward()

        with torch.no_grad():
            layer.weight.data -= step_size * layer.weight.grad
            layer.zero_grad()

        iteration += 1

        if iteration > max_iterations:
            print("Reached the maximum number of iterations without achieving the desired MSE!")
            break

    return layer
