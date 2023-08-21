import torch


def function03(x: torch.Tensor, y: torch.Tensor):
    # Fix random seed
    torch.manual_seed(0)

    # Fix n_step, step_size and maximum value of mse
    step_size = 1e-2
    max_mse = 1
    max_iterations = 5000

    # Initial mse value (to enter the loop)
    mse = float('inf')
    iteration = 0
    # Determine n_features, n_objects, weights and X
    n_features = x.shape[1]
    n_objects = y.shape[0]
    w = torch.rand(n_features, requires_grad=True, dtype=torch.float32)

    # Loop for determine mse by gradient
    while mse > 1:
        y_pred = torch.matmul(x, w) + torch.randn(n_objects) / 2
        mse = torch.mean((y_pred - y) ** 2)

        print(f'MSE {mse.item():.5f}')
        iteration += 1

        mse.backward()

        with torch.no_grad():
            w -= w.grad * step_size
            w.grad.zero_()

        if iteration > max_iterations:
            break

    return w