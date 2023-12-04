from scipy.ndimage.filters import gaussian_filter
from .base_transforms import remove_small_area
import numpy as np
import torch
from .base_transforms import fit_gauss_envelope
from skimage.morphology import convex_hull_image, binary_erosion
from scipy.ndimage import binary_erosion, generate_binary_structure


def gaussian_mask(img, factor):
    """
    Create a gaussian mask based on the image

    :param img: The image
    :param factor: Strength of the mask
    :return: The mask
    """
    *_, mask = fit_gauss_envelope(img)
    return mask ** factor


def gaussian_mask_with_info(img, factor):
    """
    Create a gaussian mask based on the image

    :param img: The image
    :param factor: Strength of the mask
    :return: The mask, the mean, the covariance matrix
    """
    mu, cov, mask = fit_gauss_envelope(img)
    mu_x = mu[0]
    mu_y = mu[1]
    cov_x = cov[0, 0]
    cov_y = cov[1, 1]
    cov_xy = cov[0, 1]
    mei_mask = mask ** factor
    return mei_mask, mu_x, mu_y, cov_x, cov_y, cov_xy


def mei_mask(img, delta_thr=0.5, size_thr=50, expansion_sigma=3, expansion_thr=0.3, filter_sigma=2):
    """
    Create a mask for the MEI image

    :param img: The image
    :param delta_thr: The threshold for the delta
    :param size_thr: The threshold for the size
    :param expansion_sigma: The sigma for the expansion
    :param expansion_thr: The threshold for the expansion
    :param filter_sigma: The sigma for the filter
    :return: The mask
    """
    img = img.squeeze()
    delta = img - img.mean()
    mask = np.abs(delta) > delta_thr
    # remove small lobes - likely an artifact
    mask = remove_small_area(mask, size_threshold=size_thr)
    # fill in the gap between lobes
    mask = convex_hull_image(mask)
    # expand the size of the mask
    mask = gaussian_filter(mask.astype(float), sigma=expansion_sigma) > expansion_thr
    # blur the edge, giving smooth transition
    mask = gaussian_filter(mask.astype(float), sigma=filter_sigma)
    return mask


def mei_tight_mask(img, operation, device, stdev_size_thr=1, filter_sigma=1, target_reduction_ratio=0.9):
    """
    Create a tight mask for the MEI image

    :param img: The image
    :param operation: The operation of the MEI
    :param device: The device
    :param stdev_size_thr: The threshold for the standard deviation
    :param filter_sigma: The sigma for the filter
    :param target_reduction_ratio: The target reduction ratio
    :return: The mask, the reduction ratio
    """
    def get_activation(mei):
        with torch.no_grad():
            img = torch.Tensor(mei).to(device)
            activation = operation(img).data.cpu().numpy()
        return activation

    img = img.squeeze()
    delta = img - img.mean()
    fluc = np.abs(delta)
    thr = np.std(fluc) * stdev_size_thr

    # original mask
    mask = convex_hull_image((fluc > thr).astype(float))
    fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
    masked_img = fm * img + (1 - fm) * img.mean()
    activation = base_line = get_activation(masked_img)

    # TODO: fix for multiple channels
    count = 0
    while activation > base_line * target_reduction_ratio:
        selem_size = 3
        selem = generate_binary_structure(img.ndim, 1)
        selem[selem_size//2] = 1
        mask = binary_erosion(mask, selem)
        fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
        masked_img = fm * img + (1 - fm) * img.mean()
        activation = get_activation(masked_img)
        count += 1

        if count > 100:
            print('This has been going on for too long! - aborting')
            raise ValueError('The activation does not reduce for the given setting')

    reduction_ratio = activation / base_line
    return fm, reduction_ratio


def mask_image(img, mask='gaussian', background=0, operation=lambda x: x, device='cpu', **MaskParams):
    """
    Applies the mask `mask` onto the `img`. The completely masked area is then
    replaced with the value `background`.

    :param img: image to be masked
    :param mask: type of mask to be applied
    :param background: value to be used for the masked area
    :param operation: operation to be applied on the masked image
    :param device: device to be used for the operation
    :param MaskParams: parameters for the mask
    :return: masked image
    """
    if mask is None:
        return img
    elif mask == 'gaussian':
        factor = MaskParams.get('factor', 1/4)
        _mask = gaussian_mask(img, factor)
    elif mask == 'meitorch':
        _mask = mei_mask(img, **MaskParams)
    elif mask == 'mei_tight':
        _mask, _ = mei_tight_mask(img, operation, device, **MaskParams)
    else:
        raise ValueError(f'Unknown mask type: {mask}')

    filler = np.full_like(img, background)
    return img * _mask + filler * (1 - _mask)
