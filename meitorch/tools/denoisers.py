import numpy as np
from numpy.linalg import svd
from skimage.restoration import denoise_nl_means, estimate_sigma

import torch
import kornia
from torch import nn
from abc import ABC, abstractmethod
from meitorch.tools.schedules import Scheduler, ConstantSchedule


class Denoiser(nn.Module, ABC):
    @abstractmethod
    def forward(self, noisy_image, step_i):
        pass

    @staticmethod
    def init_p(parameter):
        if isinstance(parameter, Scheduler):
            return parameter
        else:
            return ConstantSchedule(parameter)

    def get_denoiser(self, type, **params):
        if type == 'bilateral':
            denoiser = FilterBlur(type=type, **params)
        elif type == 'gaussian':
            denoiser = FilterBlur(type=type, **params)
        elif type == 'tv':
            assert 'regularization_scaler' in params.keys(), "regularization_scaler must be specified for tv denoiser"
            assert 'num_iters' in params.keys(), "num_iters must be specified for tv denoiser"
            assert 'lr' in params.keys(), "lr must be specified for tv denoiser"
            denoiser = TVDenoise(**params)
        elif type == 'wnnm':
            assert 'patch_size' in params.keys(), "patch_size must be specified for wnnm denoiser"
            assert 'delta' in params.keys(), "delta must be specified for wnnm denoiser"
            assert 'c' in params.keys(), "c must be specified for wnnm denoiser"
            assert 'K' in params.keys(), "K must be specified for wnnm denoiser"
            assert 'sigma_n' in params.keys(), "sigma_n must be specified for wnnm denoiser"
            assert 'N_threshold' in params.keys(), "N_threshold must be specified for wnnm denoiser"
            assert 'N_iter' in params.keys(), "N_iter must be specified for wnnm denoiser"
            denoiser = WNNMDenoiser(**params)
        elif type == 'bm3d':
            assert 'sigma_psd' in params.keys(), "sigma_psd must be specified for bm3d denoiser"
            denoiser = BM3DDenoise(**params)
        elif type == 'nlm':
            assert 'patch_size' in params.keys(), "patch_size must be specified for nlm denoiser"
            assert 'patch_distance' in params.keys(), "patch_distance must be specified for nlm denoiser"
            assert 'h' in params.keys(), "h must be specified for nlm denoiser"
            assert 'fast_mode' in params.keys(), "fast_mode must be specified for nlm denoiser"
            denoiser = NonLocalMeansDenoise(**params)
        else:
            raise ValueError("Invalid denoiser type "
                             "(must be 'bilateral', 'tv', 'filter', 'wnnm', 'bm3d')")
        return denoiser


class TVDenoise(Denoiser):
    def __init__(self, regularization_scaler=0.0001, num_iters=500, lr=0.1):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()

        self.regularization_scaler = self.init_p(regularization_scaler)
        self.num_iters = self.init_p(num_iters)
        self.lr = self.init_p(lr)

    def loss(self, clean_image, noisy_image, step_i):
        return self.l2_term(clean_image, noisy_image) + \
            self.regularization_scaler(step_i) * self.regularization_term(clean_image)

    def forward(self, noisy_image, step_i):
        clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([clean_image], lr=self.lr(step_i))
        for i in range(self.num_iters(step_i)):
            optimizer.zero_grad()
            loss = self.loss(clean_image, noisy_image, step_i)
            loss.backward()
            optimizer.step()
        return self.clean_image


class FilterBlur(Denoiser):
    def __init__(self, type, **filter_params):
        super().__init__()
        from kornia import filters

        self.type = type

        if type == 'bilateral':
            assert 'kernel_size' in filter_params.keys(), "kernel_size must be specified for bilateral filter"
            assert 'sigma_color' in filter_params.keys(), "sigma_color must be specified for bilateral filter"
            assert 'sigma_spatial' in filter_params.keys(), "sigma_spatial must be specified for bilateral filter"
            params = lambda step_i: (filter_params['kernel_size'],
                                     self.init_p(filter_params['sigma_color'])(step_i),
                                     self.init_p(filter_params['sigma_spatial'])(step_i))
            filter = filters.BilateralBlur
        elif type == 'gaussian':
            assert 'kernel_size' in filter_params.keys(), "kernel_size must be specified for gaussian filter"
            assert 'sigma' in filter_params.keys(), "sigma must be specified for gaussian filter"
            params = lambda step_i: (filter_params['kernel_size'],
                                     self.init_p(filter_params['sigma'])(step_i))
            filter = filters.GaussianBlur2d
        else:
            raise ValueError("Invalid filter type "
                             "(must be 'bilateral', 'gaussian')")

        self.filter = filter
        self.params = params

    def forward(self, x, step_i):
        params = self.params(step_i)
        return self.filter(*params)(x)


class NonLocalMeansDenoise(Denoiser):
    def __init__(self, patch_size=5, patch_distance=6, h=0.8, fast_mode=True):
        super().__init__()

        self.patch_size = self.init_p(patch_size)
        self.patch_distance = self.init_p(patch_distance)
        self.h = self.init_p(h)
        self.fast_mode = fast_mode

    def forward(self, noisy_img, step_i):
        # estimate the noise standard deviation from the noisy image
        sigma_est = np.mean(estimate_sigma(noisy_img, channel_axis=1))
        print(f'estimated noise standard deviation = {sigma_est}')

        patch_kw = dict(patch_size=self.patch_size(step_i),
                        patch_distance=self.patch_distance(step_i),  # 13x13 search area
                        channel_axis=1)
        denoised = denoise_nl_means(noisy_img, h=self.h(step_i) * sigma_est,
                                    fast_mode=self.fast_mode, **patch_kw)
        return denoised


class WNNMDenoiser(Denoiser):
    """
     Applies weighted nuclear norm minimization based denoising to the imput image img
     TODO: change channel dimension (now its on -1 implicitly)
    """

    def __init__(self, patch_size, delta, c, K, sigma_n, N_threshold, N_iter=3):
        super().__init__()
        self.patch_size = self.init_p(patch_size)
        self.delta = self.init_p(delta)
        self.c = self.init_p(c)
        self.K = self.init_p(K)
        self.sigma_n = self.init_p(sigma_n)
        self.N_threshold = self.init_p(N_threshold)
        self.N_iter = self.init_p(N_iter)  # the number of iterations for estimating \hat{X}_j

    def forward(self, noisy_img, step_i):
        patch_size = self.patch_size(step_i)

        # Specify the search window
        searchWindowRadius = patch_size * 3

        # Specify the width of padding and pad the noisy image
        pad = searchWindowRadius + patch_size
        imgPad = np.pad(noisy_img, pad_width=pad)
        imgPad = imgPad[..., pad:-pad]

        # Initialize variables to be iterated over
        xhat_iter = noisy_img

        for n in range(self.K(step_i)):
            # Pad the image for the iteration
            xhat_iter = np.pad(xhat_iter, pad_width=pad)
            xhat_iter = xhat_iter[..., pad:-pad]

            # Regularize the image that is denoised during the iteration
            y_iter = xhat_iter + self.delta(step_i) * (imgPad - xhat_iter)

            # Initialize the matrix to keep track of how many times each pixel has been updated
            pixel_contribution_matrix = np.ones_like(imgPad)

            # Identify similar patches and produce the matrix of similar patches
            for j in range(noisy_img.shape[0]):
                for i in range(noisy_img.shape[1]):
                    # Select the central patch
                    centerPatch = \
                        y_iter[j + searchWindowRadius:j + searchWindowRadius + 2 * patch_size,
                        i + searchWindowRadius:i + searchWindowRadius + 2 * patch_size
                        :]

                    # Initialize the vector of distances between patches
                    dists = np.ones(((2 * searchWindowRadius + 1) ** 2))
                    # Initialize the matrix of patches
                    patches = np.zeros(((2 * searchWindowRadius + 1) ** 2, (2 * patch_size) ** 2))
                    # Compute distances between patches
                    # This is partially vectorized by using indexing to take out patches in a sliding window fashing
                    # out of a vertical slice through the search window
                    for k in range(2 * searchWindowRadius + 1):
                        # Take a vertical slice in the search window
                        otherPatch = y_iter[j:j + 2 * pad,
                                     i + k:i + k + 2 * patch_size,
                                     :]

                        # Determine indices corresponding to patches in a window sliding down the search window
                        indexer = np.arange((2 * patch_size) ** 2)[None, :] + (2 * patch_size) * np.arange(
                            otherPatch.shape[0] - 2 * patch_size + 1)[:, None]

                        # Set columns to be patches
                        otherPatch = otherPatch.flatten()
                        otherPatch = np.reshape(otherPatch[indexer],
                                                (otherPatch[indexer].shape[0], (2 * patch_size) ** 2))

                        # Compute distance and store the corresponding patches
                        dists[k * (2 * searchWindowRadius + 1):(k + 1) * (2 * searchWindowRadius + 1)] = (
                                np.sum((centerPatch.reshape(((2 * patch_size) ** 2)) - otherPatch) ** 2,
                                       axis=1) / (2 * patch_size) ** 2).flatten()
                        patches[k * (2 * searchWindowRadius + 1):(k + 1) * (2 * searchWindowRadius + 1), :] = otherPatch

                    # Select to N_threshold nearest patches and creat a patch matrix
                    indcs = np.argsort(dists)
                    Yj = (patches[indcs[:self.N_threshold(step_i)], :]).transpose()

                    # Center the columns
                    Yj_means = np.sum(Yj, axis=0)
                    Yj_center = Yj - Yj_means

                    # First iteration need to estimate singular values of Xj
                    U, S, V_T = svd(Yj_center, full_matrices=False)
                    sing_val = np.sqrt(np.maximum(S ** 2 - self.N_threshold(step_i) * self.sigma_n(step_i) ** 2, 0))

                    # Calculate the weights and sinfular values of \hat{X}_j iteratively
                    for m in range(self.N_iter(step_i)):
                        w = self.c(step_i) * np.sqrt(self.N_threshold(step_i)) / (sing_val + 10 ** (-6))
                        sing_val = np.diag(np.maximum(S - w, 0))

                    # Compute \hat{X}_j
                    Xj_hat_center = U @ np.diag(np.maximum(S - w, 0)) @ V_T
                    Xj_hat = Xj_hat_center + Yj_means

                    # Add the estimate of denoised central patch (first column of \hat{X}_j) to the esmated denoised image clipping it to between 0 and 1
                    xhat_iter[j + searchWindowRadius:j + searchWindowRadius + 2 * patch_size,
                    i + searchWindowRadius:i + searchWindowRadius + 2 * patch_size,
                    :] = xhat_iter[j + searchWindowRadius:j + searchWindowRadius + 2 * patch_size,
                         i + searchWindowRadius:i + searchWindowRadius + 2 * patch_size,
                         :] + np.clip(Xj_hat[:, 0].reshape((2 * patch_size, 2 * patch_size, 1)), 0, 1)

                    # Keep track of how many times each pixel has been added to
                    pixel_contribution_matrix[j + searchWindowRadius:j + searchWindowRadius + 2 * patch_size,
                    i + searchWindowRadius:i + searchWindowRadius + 2 * patch_size,
                    :] = pixel_contribution_matrix[j + searchWindowRadius:j + searchWindowRadius + 2 * patch_size,
                         i + searchWindowRadius:i + searchWindowRadius + 2 * patch_size,
                         :] + np.ones_like(
                        pixel_contribution_matrix[j + searchWindowRadius:j + searchWindowRadius + 2 * patch_size,
                        i + searchWindowRadius:i + searchWindowRadius + 2 * patch_size, :])

            # Remove the padding and average out contributions to pixels from different patches
            xhat_iter = xhat_iter[pad:-pad, pad:-pad, :] / pixel_contribution_matrix[pad:-pad, pad:-pad, :]

        # Produce the final output
        out = xhat_iter
        return out


class BM3DDenoise(Denoiser):
    def __init__(self, sigma_psd=30 / 255):
        super().__init__()
        self.sigma_psd = self.init_p(sigma_psd)

    def forward(self, noisy_img, step_i):
        import bm3d
        denoised_image = bm3d.bm3d(noisy_img,
                                   sigma_psd=self.sigma_psd(step_i),
                                   stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        return denoised_image
