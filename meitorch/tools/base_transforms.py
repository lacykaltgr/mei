import warnings
from tqdm import tqdm

import numpy as np
import torch
from scipy.ndimage import label
from numpy.linalg import inv, cholesky


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

    g = gauss2d(vx, vy, (mu_x, mu_y), cov)
    mu = (mu_x, mu_y)
    return mu, cov, np.sqrt(g.reshape(img.shape))


def roll(tensor, shift, axis):
    """
    Rolls the tensor along the given axis by the given shift

    :param tensor: The tensor to roll
    :param shift: The shift to apply
    :param axis: The axis on which to shift
    :return: The shifted tensor
    """
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


def remove_small_area(mask, size_threshold=50):
    """
    Removes contiguous areas in a thresholded image that is smaller in the number of pixels than size_threshold.

    :param mask: The thresholded image
    :param size_threshold: The threshold
    :return: The thresholded image with small areas removed
    """
    mask_mod = mask.copy()
    label_im, nb_labels = label(mask_mod)
    for v in range(0, nb_labels+1):
        area = label_im == v
        s = np.sum(area)
        if s < size_threshold:
            mask_mod[area] = 0
    return mask_mod





def adjust_img_stats(img, mu, sigma, img_min=0, img_max=255, mask=None, max_gain=6000, min_gain=0.001,
                     base_ratio=1.05, verbose=False):
    """
    Adjusts the image statistics to the given mean and standard deviation.

    :param img: The image to adjust
    :param mu: The target mean
    :param sigma: The target standard deviation
    :param img_min: The minimum pixel intensity allowed
    :param img_max: The maximum pixel intensity allowed
    :param mask: The mask to use
    :param max_gain: The maximum gain to use
    :param min_gain: The minimum gain to use
    :param base_ratio: The base ratio to use
    :param verbose: The verbosity
    :return: The adjusted image
    """
    if mask is None:
        mask = np.ones_like(img)
    mimg = img * mask

    delta = img - mimg.sum() / mask.sum()

    def get_image(delta, offset=0):
        return np.clip((delta * mask) + mu + offset, img_min, img_max)

    sigma_i = get_image(delta).std()
    if sigma_i < 1e-8:
        warnings.warn('Zero standard deviation detected.')
        img = np.clip((delta * mask) + mu, img_min, img_max)  # flat image
        unmasked_img = np.clip(delta + mu, img_min, img_max)
    else:
        gain = sigma / sigma_i

        dir_sign = np.sign(np.log(gain))
        ratio = base_ratio ** dir_sign

        max_gain = min(max(img_max - mu, mu - img_min) / np.min(np.abs(delta)[mask > 0]), max_gain)
        min_gain = min_gain

        v = np.linspace(-30, 30, 1000)[:, None, None]

        imgs = []
        unmasked_imgs = []
        conts = []
        while True:
            if verbose:
                print('Trying gain', gain)
            adj_images = get_image(delta * gain, v)
            pos = np.argmin(np.abs(adj_images.mean(axis=(1, 2)) - mu))
            img = adj_images[pos]
            unmasked_img = np.clip(delta * gain + mu + v[pos], img_min, img_max)

            cont = img.std()
            if verbose:
                print('Got mean and contrast', img.mean(), cont)
            imgs.append(img)
            conts.append(cont)
            unmasked_imgs.append(unmasked_img)
            if (cont - sigma) * dir_sign > 0:
                break
            gain = gain * ratio
            if gain > max_gain or gain < min_gain:
                break

        pos = np.argmin(np.abs(np.array(conts) - sigma))
        img = imgs[pos]
        unmasked_img = unmasked_imgs[pos]
    if verbose:
        print('Selected version with mu={} and std={}'.format(img.mean(), img.std()))
    return img, unmasked_img


def adjust_contrast(img, contrast=-1, mu=-1, img_min=0, img_max=255, force=True, verbose=False, steps=5000):
    """
    Performs contrast adjustment of the image, being mindful of the image value bounds (e.g. [0, 255]). Given the bounds
    the normal shift and scale will not guarantee that the resulting image still has the desired mean luminance
    and contrast.

    :param img: The image to adjust the contrast of
    :param contrast: The desired contrast (RMS)
    :param mu: The desired mean luminance
    :param img_min: The minimum pixel intensity allowed
    :param img_max: The maximum pixel intensity allowed
    :param force: if True, iterative approach is taken to produce an image with the desired stats. This will likely cause
                some pixels to saturate in the upper and lower bounds. If False, then image is scaled simply based on ratio of
                current and desired contrast, and then clipped. This likely results in an image that is higher in contrast
                than the original but not quite at the desired contrast and with some pixel information lost due to clipping.
    :param verbose: If True, prints out progress during iterative adjustment
    :param steps: If force=True, this sets the number of iterative steps to be used in adjusting the image. The larger the
                value, the closer the final image would approach the desired contrast.

    :return:
        adjusted_image - a new image adjusted from the original such that the desired mean/contrast is achieved to the
            best of the configuration.
        clipped - Whether any clipping took place. If True, it indicates that some clipping of pixel intensities occured
            and thus some pixel information was lost.
        actual_contrast - the final contrast of the image reached
    """
    current_contrast = img.std()

    if contrast < 0:
        gain = 1   # no adjustment of contrast
    else:
        gain = contrast / current_contrast

    delta = img - img.mean()
    if mu is None or mu < 0: # no adjustment of mean
        mu = img.mean()

    min_pdist = delta[delta > 0].min()
    min_ndist = (-delta[delta < 0]).min()

    # point beyond which scaling would completely saturate out the image (e.g. all pixels would be completely
    # black or white)
    max_lim_gain = max((img_max - mu) / min_pdist, (mu - img_min) / min_ndist)

    vmax = delta.max()
    vmin = delta.min()

    # maximum gain that could be used without losing image information
    max_gain = min((img_max - mu) / vmax, (img_min-mu) / vmin)

    # if True, we already know that the desired contrast cannot be achieved without losing some pixel information
    # into the saturation regime
    clipped = gain > max_gain

    v = np.linspace(0, (img_max-img_min), steps) # candidates for mean adjustment
    if clipped and force:
        if verbose:
            print('Adjusting...')
        cont = []
        imgs = []
        gains = np.logspace(np.log10(gain), np.log10(max_lim_gain), steps)
        # for each gain, perform offset adjustment such that the mean is equal to the set value
        for g in gains:
            img = delta * g + mu
            img = np.clip(img, img_min, img_max)
            offset = img.mean() - mu # shift in clipped image mean caused by the clipping
            if offset < 0: # pixel values needs to be raised
                offset = -offset
                mask = (img_max-img < v[:, None, None])
                nlow = mask.sum(axis=(1, 2)) # pixels that are closer to the bound than v
                nhigh = img.size - nlow
                # calculate the actual shift in mean that can be achieved by shifting all pixels by v
                # then clipping
                va = ((mask * (img_max-img)).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)

                # find the best candidate offset that achieves closest to the desired shift in the mean
                pos = np.argmin(np.abs(va - offset))
                actual_offset = -v[pos]
            else:
                mask = (img-img_min < v[:, None, None])
                nlow = mask.sum(axis=(1, 2))
                nhigh = img.size - nlow
                # actual shift in mean that can be achieved by shifting all pixels by v
                va = ((mask * (img-img_min)).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)
                pos = np.argmin(np.abs(va - offset))
                actual_offset = v[pos]


            img = img - actual_offset
            img = np.clip(img, img_min, img_max)
            # actual contrast achieved with this scale and adjustment
            c = img.std()
            cont.append(c)
            imgs.append(img)
            if c > contrast:
                break
        loc = np.argmin(np.abs(np.array(cont) - contrast))
        adj_img = imgs[loc]
    else:
        adj_img = delta * gain + mu
        adj_img = np.clip(adj_img, img_min, img_max)
    actual_contrast = adj_img.std()
    return adj_img, clipped, actual_contrast


def adjust_contrast_with_mask(img, img_mask=None, contrast=-1, mu=-1, img_min=0, img_max=255, force=True, verbose=False,
                              mu_steps=500, gain_steps=500):
    """
    A version of the contrast adjustment that is mindful of the mask

    Performs contrast adjustment of the image, being mindful of the image value bounds (e.g. [0, 255]). Given the bounds
    the normal shift and scale will not guarantee that the resulting image still has the desired mean luminance
    and contrast.

    :param img: The image to adjust the contrast of
    :param img_mask: The mask to use for the adjustment. If None, then the entire image is used.
    :param contrast: The desired contrast (RMS)
    :param mu: The desired mean luminance
    :param img_min: The minimum pixel intensity allowed
    :param img_max: The maximum pixel intensity allowed
    :param force: if True, iterative approach is taken to produce an image with the desired stats. This will likely cause
                some pixels to saturate in the upper and lower bounds. If False, then image is scaled simply based on ratio of
                current and desired contrast, and then clipped. This likely results in an image that is higher in contrast
                than the original but not quite at the desired contrast and with some pixel information lost due to clipping.
    :param verbose: If True, prints out progress during iterative adjustment
    :param gain_steps: If force=True, this sets the number of iterative steps to be used in adjusting the image. The larger the
                value, the closer the final image would approach the desired contrast.

    :return:
        adjusted_image - a new image adjusted from the original such that the desired mean/contrast is achieved to the
            best of the configuration.
        clipped - Whether any clipping took place. If True, it indicates that some clipping of pixel intensities occured
            and thus some pixel information was lost.
        actual_contrast - the final contrast of the image reached
    """
    if img_mask is None:
        img_mask = np.ones_like(img)

    def get_mu(x):
        return np.sum(img_mask * x) / np.sum(img_mask)

    def get_sigma(x):
        h, w = x.shape[-2:]
        avg = get_mu(x)
        return np.sqrt(np.sum(img_mask ** 2 * (x - avg) ** 2) / (h * w))

    adj_img = img * img_mask + mu * (1 - img_mask) #adj_img volt
    adj_img = np.clip(adj_img, img_min, img_max)
    mimg = img_mask * img
    test_img = np.clip(mimg - mimg.mean() + mu, img_min, img_max)
    current_contrast = test_img.std()
    if verbose:
        print('Initial contrast:', current_contrast)

    if contrast < 0:
        gain = 1  # no adjustment of contrast
    else:
        gain = contrast / current_contrast

    delta = (img - get_mu(img))  # * bin_mask # only consider deltas in mask region
    if mu is None or mu < 0:  # no adjustment of mean
        mu = get_mu(img)

    min_pdist = delta[delta > 0].min()
    min_ndist = (-delta[delta < 0]).min()

    # point beyond which scaling would completely saturate out the image (e.g. all pixels would be completely black or
    # white)
    max_lim_gain = min(max((img_max - mu) / min_pdist, (mu - img_min) / min_ndist), 100)

    vmax = (delta * img_mask).max()
    vmin = (delta * img_mask).min()

    # maximum gain that could be used without losing image information
    max_gain = min((img_max - mu) / vmax, (img_min - mu) / vmin)


    # if True, we already know that the desired contrast cannot be achieved without losing some pixel information
    # into the saturation regime
    clipped = gain > max_gain
    print('gains', gain , max_gain)

    v = np.linspace(0, (img_max - img_min), mu_steps)  # candidates for mean adjustment

    if clipped and force:
        if verbose:
            print('Adjusting...')
        cont = []
        imgs = []
        gains = np.logspace(np.log10(gain), np.log10(max_lim_gain), gain_steps)
        # for each gain, perform offset adjustment such that the mean is equal to the set value
        for g in tqdm(gains, disable=(not verbose)):
            print('')
            img = delta * g + mu
            img = np.clip(img, img_min, img_max)

            offset = mu - get_mu(img)  # shift in clipped image mean caused by the clipping
            if offset > 0:
                sign = 1
                edge = img_max
            else:
                sign = -1
                edge = img_min

            offset = sign * offset
            mask = (sign * (edge - img) < v[:, None, None])

            nlow = (mask * img_mask).sum(axis=(1, 2))  # effective number of pixels that are closer to the bound than v
            nhigh = img_mask.sum() - nlow

            # calculate the actual shift in mean that can be achieved by shifting all pixels by v
            # then clipping
            va = ((mask * img_mask * sign * (edge - img)).sum(axis=(1, 2)) + (
                    v[:, None, None] * img_mask * (1 - mask)).sum(axis=(1, 2))) / (nlow + nhigh)

            # find the best candidate offset that achieves closest to the desired shift in the mean
            pos = np.argmin(np.abs(va - offset))
            actual_offset = sign * v[pos]

            img = img + actual_offset
            img = np.clip(img, img_min, img_max)
            # actual contrast achieved with this scale and adjustment
            c = get_sigma(img)
            print('contrast now', c)
            cont.append(c)
            imgs.append(img)
            if c > contrast:
                break
        loc = np.argmin(np.abs(np.array(cont) - contrast))
        adj_img = imgs[loc]
    else:
        adj_img = delta * gain + mu

    adj_img = adj_img * img_mask + mu * (1 - img_mask)
    adj_img = np.clip(adj_img, img_min, img_max)
    actual_contrast = adj_img.std()
    return adj_img, clipped, actual_contrast