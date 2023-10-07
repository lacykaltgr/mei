import torch
import numpy as np


def fft_smooth(grad, factor=1/4):
    """
    Tones down the gradient with 1/sqrt(f) filter in the Fourier domain.
    Equivalent to low-pass filtering in the spatial domain.

    :param grad: The gradient
    :param factor: The factor
    """
    if factor == 0:
        return grad
    h, w = grad.size()[-2:]
    tw = np.minimum(np.arange(0, w), np.arange(w-1, -1, -1), dtype=np.float32)  # [-(w+2)//2:]
    th = np.minimum(np.arange(0, h), np.arange(h-1, -1, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** factor)
    F = grad.new_tensor(t / t.mean()).unsqueeze(-1)
    pp = torch.fft.rfft(grad.data, 2)
    return torch.fft.irfft(pp * F, 2)