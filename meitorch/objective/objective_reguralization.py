import kornia
import torch


def tv_norm(image):
    """
    # Calculate the TV norm
    # TV(x) = ||∇x||_1, where ∇x is the gradient of the image
    # You can use finite differences to calculate the gradient
    dx = torch.abs(image[:, :-1] - image[:, 1:])  # Differences along columns
    dy = torch.abs(image[:-1, :] - image[1:, :])  # Differences along rows

    # Sum the absolute differences to calculate the TV norm
    tv_norm = dx.sum() + dy.sum()
    """

    loss = kornia.losses.TotalVariation()
    return loss(image)


def l2_norm(image):
    return torch.norm(image, p=2)
