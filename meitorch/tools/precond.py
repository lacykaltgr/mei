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
    tw = np.minimum(np.arange(0, w), np.arange(w, 0, -1), dtype=np.float32)
    th = np.minimum(np.arange(0, h), np.arange(h, 0, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** factor)
    F = grad.new_tensor(t / t.mean()).unsqueeze(0)
    pp = torch.fft.fft2(grad.data, dim=(-1, -2))
    return torch.real(torch.fft.ifft2(pp * F, dim=(-1, -2)))
