is_cuda = lambda m: next(m.parameters()).is_cuda
import torch
import numpy as np
from numpy.linalg import inv, cholesky
#from scipy import ndimage
from itertools import product, zip_longest, count

def fft_smooth(grad, factor=1/4):
    """
    Tones down the gradient with 1/sqrt(f) filter in the Fourier domain.
    Equivalent to low-pass filtering in the spatial domain.
    """
    if factor == 0:
        return grad
    #h, w = grad.size()[-2:]
    # grad = tf.transpose(grad, [0, 3, 1, 2])
    # grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    h, w = grad.size()[-2:]
    # grad = tf.transpose(grad, [0, 3, 1, 2])
    # grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    tw = np.minimum(np.arange(0, w), np.arange(w-1, -1, -1), dtype=np.float32)  # [-(w+2)//2:]
    th = np.minimum(np.arange(0, h), np.arange(h-1, -1, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** factor)
    F = grad.new_tensor(t / t.mean()).unsqueeze(-1)
    print(F.shape)
    pp = torch.fft.rfft(grad.data, 2)
    print(pp.shape)
    return torch.fft.irfft(pp * F, 2)


def blur(img, sigma):
    if sigma > 0:
        for d in range(len(img)):
            pass
            #img[d] = ndimage.filters.gaussian_filter(img[d], sigma, order=0)
    return img


def blur_in_place(tensor, sigma):
    blurred = np.stack([blur(im, sigma) for im in tensor.cpu().numpy()])
    tensor.copy_(torch.Tensor(blurred))




def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)



def batch_mean(batch, keepdim=False):
    """ Compute mean for a batch of images. """
    mean = batch.view(len(batch), -1).mean(-1)
    if keepdim:
        mean = mean.view(len(batch), 1, 1, 1)
    return mean


def batch_std(batch, keepdim=False, unbiased=True):
    """ Compute std for a batch of images. """
    std = batch.view(len(batch), -1).std(-1, unbiased=unbiased)
    if keepdim:
        std = std.view(len(batch), 1, 1, 1)
    return std


def query(x, query):
    for i in range(len(query)):
        x = x[:, query[i]]
    return x

def gauss2d(vx, vy, mu, cov):
    input_shape = vx.shape
    mu_x, mu_y = mu
    v = np.stack([vx.ravel() - mu_x, vy.ravel() - mu_y])
    cinv = inv(cholesky(cov))
    y = cinv @ v
    g = np.exp(-0.5 * (y * y).sum(axis=0))
    return g.reshape(input_shape)

def fit_gauss_envelope(img):
    """
    Given an image, finds a Gaussian fit to the image by treating the square of mean shifted image as the distribution.
    Args:
        img:

    Returns:

    """
    vx, vy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    rect = (img - img.mean()) ** 2
    pdf = rect / rect.sum()
    mu_x = (vx * pdf).sum()
    mu_y = (vy * pdf).sum()

    cov_xy = (vx * vy * pdf).sum() - mu_x * mu_y
    cov_xx = (vx ** 2 * pdf).sum() - mu_x ** 2
    cov_yy = (vy ** 2 * pdf).sum() - mu_y ** 2

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    g = gauss2d(vx, vy, (mu_x, mu_y), cov)
    mu = (mu_x, mu_y)
    return mu, cov, np.sqrt(g.reshape(img.shape))


def remove_small_area(mask, size_threshold=50):
    """
    Removes contiguous areas in a thresholded image that is smaller in the number of pixels than size_threshold.
    """
    mask_mod = mask.copy()
    label_im, nb_labels = None#label(mask_mod)
    for v in range(0, nb_labels+1):
        area = label_im == v
        s = np.sum(area)
        if s < size_threshold:
            mask_mod[area] = 0
    return mask_mod


def adj_model(models, neuron_query):
    def adj_model_fn(x):
        count = 0
        sum = None
        for model in models:
            y = query(model(x), neuron_query)
            sum = y if count == 0 else sum + y
            count += 1
        return sum / count
    return adj_model_fn
