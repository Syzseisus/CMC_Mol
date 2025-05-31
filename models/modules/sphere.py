import torch


def unit_sphere_(tensor: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    In-place initialize the input tensor with random unit vectors on a hypersphere scaled by alpha.

    Args:
        tensor  : Tensor of shape (..., N) to initialize in-place.
        alpha   : Scale factor to apply to each unit vector (default: 0.01).

    Returns:
        The same tensor, now filled with scaled unit vectors.
    """
    rand = torch.randn_like(tensor)  # shape (..., N)
    lengths = rand.norm(dim=-1, keepdim=True)  # shape (..., 1)
    unit = rand / (lengths + 1e-8)  # shape (..., N)

    with torch.no_grad():
        tensor.copy_(unit * alpha)
    return tensor
