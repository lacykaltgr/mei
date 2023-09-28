import numpy as np
from skimage.morphology import convex_hull_image
from scipy.ndimage import binary_erosion, generate_binary_structure
import scipy
from numpy.linalg import inv, cholesky
import torch


def wiener_filter(input_signal, noise_power, signal_power):
    # Wiener filter implementation
    gain = signal_power / (signal_power + noise_power)
    denoised_signal = gain * input_signal
    return denoised_signal


class Gaussian:
    @staticmethod
    def gauss2d(vx, vy, mu, cov):
        """
        Computes the 2D Gaussian distribution with the given parameters.

        :param vx: x coordinate
        :param vy: y coordinate
        :param mu: mean
        :param cov: covariance
        :return: The Gaussian distribution
        """
        input_shape = vx.shape
        mu_x, mu_y = mu
        v = np.stack([vx.ravel() - mu_x, vy.ravel() - mu_y])
        cinv = inv(cholesky(cov))
        y = cinv @ v
        g = np.exp(-0.5 * (y * y).sum(axis=0))
        return g.reshape(input_shape)

    @staticmethod
    def fit_gauss_envelope(img):
        """
        Given an image, finds a Gaussian fit to the image by treating the square of mean shifted image as the distribution.

        :param img: The image
        :return: The Gaussian distribution
        """
        # scale down the image to 2 dimensions
        while len(img.shape) > 2:
            img = img.mean(axis=0)
        vx, vy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        rect = (img - img.mean()) ** 2
        pdf = rect / rect.sum()
        mu_x = (vx * pdf).sum()
        mu_y = (vy * pdf).sum()

        cov_xy = (vx * vy * pdf).sum() - mu_x * mu_y
        cov_xx = (vx ** 2 * pdf).sum() - mu_x ** 2
        cov_yy = (vy ** 2 * pdf).sum() - mu_y ** 2

        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

        g = Gaussian.gauss2d(vx, vy, (mu_x, mu_y), cov)
        mu = (mu_x, mu_y)
        return mu, cov, np.sqrt(g.reshape(img.shape))

    @staticmethod
    def blur(img, sigma):
        """
        Blurs the image with a Gaussian filter

        :param img: The image
        :param sigma: The sigma of the blur
        :return: The blurred image
        """
        if sigma > 0:
            for d in range(len(img)):
                img[d] = scipy.ndimage.filters.gaussian_filter(img[d], sigma, order=0)
        return img

    @staticmethod
    def blur_in_place(tensor, sigma):
        """
        Blurs the image with a Gaussian filter IN PLACE

        :param tensor: The tensor for the image
        :param sigma: The sigma of the blur
        :return: The blurred image
        """
        blurred = np.stack([Gaussian.blur(im, sigma) for im in tensor.cpu().numpy()])
        tensor.copy_(torch.Tensor(blurred))