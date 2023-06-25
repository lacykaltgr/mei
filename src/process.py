import torch
import numpy as np
from tqdm import tqdm
from itertools import product

from .utils import roll, batch_std, fft_smooth, blur_in_place, mask_image
from .gabor import Gabor

# import scipy
# from scipy import signal, ndimage
# from scipy.ndimage import label


class _Process:
    def __init__(self, operation, bias=0, scale=1, device='cpu'):
        self.device = device
        self.bias = bias
        self.scale = scale
        self.image = None
        self.operation = operation

    def best_match(self, dataloader, mask=None, factor=1.0):
        img_activations = []
        for image, label in tqdm(dataloader.dataset):
            image = np.atleast_3d(mask_image(image, mask, self.bias, factor))  # ensure channel dimension exist
            image = torch.tensor(image[None, ...], dtype=torch.float32, requires_grad=True, device=self.device)
            y = self.operation(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        pos = np.argmax(img_activations)
        return img_activations[pos], dataloader.dataset[pos]

    def masked(self, mask, factor=1.0):
        if self.image is None:
            return None
        return mask_image(self.image, mask, self.bias, factor)

    # TODO: paraméterek még kellenének
    def masked_responses(self, mask='gaussian', factor=1.0):
        if self.image is None:
            return None
        return Gabor.masked_responses([self.image], mask, self.bias, factor)

    def jittered_responses(self, jitter_size):
        if self.image is None:
            return None

        # jitter_size = 0 vagy 5
        shift = list(enumerate(range(-jitter_size, jitter_size + 1)))
        activations = np.empty((len(shift), len(shift)))

        jiterred_images = []

        with torch.no_grad():
            img = torch.Tensor(self.image).to(self.device)

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                jiterred_images.append(jittered_img)
                activations[iy, ix] = self.operation(jittered_img).data.cpu().numpy()[0]

        return activations, jiterred_images

    def shifted_responses(self, x_shift, y_shift):

        if self.image is None:
            return None

        shifted_mei = np.roll(np.roll(self.image, x_shift, 1), y_shift, 0)

        with torch.no_grad():
            img = torch.Tensor(shifted_mei[..., None]).to(self.device)
            activations = self.operation(img).data.cpu().numpy()[0]

        return activations, shifted_mei


class GaborProcess(_Process):
    def __init__(self, operation=None, bias=0, scale=1, device='cpu', **GaborParams):
        super().__init__(operation, bias, scale, device)

        self.seed = GaborParams['seed'] if 'seed' in GaborParams else None
        self.activation = GaborParams['activation'] if 'activation' in GaborParams else None
        self.phase = GaborParams['phase'] if 'phase' in GaborParams else None
        self.wavelength = GaborParams['wavelength'] if 'wavelength' in GaborParams else None
        self.orientation = GaborParams['orientation'] if 'orientation' in GaborParams else None
        self.sigma = GaborParams['sigma'] if 'sigma' in GaborParams else None
        self.dy = GaborParams['dy'] if 'dy' in GaborParams else None
        self.dx = GaborParams['dx'] if 'dx' in GaborParams else None



# TODO: add support for multiple octaves
# TODO: lehessen specific operationnel (paraméterben megadva) használni
class MEIProcess(_Process):
    def __init__(self, operation, bias=0, scale=1, device='cpu', **MEIParams):
        super().__init__(operation, bias, scale, device)

        # result parameters
        self.activation = None
        self.monotonic = None
        self.max_contrast = None
        self.max_activation = None
        self.sat_contrast = None
        self.img_mean = None
        self.lim_contrast = None
        self.point_rf = None

        # mei parameters
        self.iter_n = 1000 if 'iter_n' not in MEIParams else MEIParams['iter_n']
        self.start_sigma = 1.5 if 'start_sigma' not in MEIParams else MEIParams['start_sigma']
        self.end_sigma = 0.01 if 'end_sigma' not in MEIParams else MEIParams['end_sigma']
        self.start_step_size = 3.0 if 'start_step_size' not in MEIParams else MEIParams['start_step_size']
        self.end_step_size = 0.125 if 'end_step_size' not in MEIParams else MEIParams['end_step_size']
        self.precond = 0  # .1          if 'precond' not in MEIParams else MEIParams['precond']
        self.step_gain = 0.1 if 'step_gain' not in MEIParams else MEIParams['step_gain']
        self.jitter = 0 if 'jitter' not in MEIParams else MEIParams['jitter']
        self.blur = True if 'blur' not in MEIParams else MEIParams['blur']
        self.norm = -1 if 'norm' not in MEIParams else MEIParams['norm']
        self.train_norm = -1 if 'train_norm' not in MEIParams else MEIParams['train_norm']
        self.clip = True if 'clip' not in MEIParams else MEIParams['clip']

        self.octaves = [
            {
                'iter_n': self.iter_n,
                'start_sigma': self.start_sigma,
                'end_sigma': self.end_sigma,
                'start_step_size': self.start_step_size,
                'end_step_size': self.end_step_size,
            },
        ]



    def make_step(self, src, step_size=1.5, sigma=None, eps=1e-12, add_loss=0):
        """
        Update src in place making a gradient ascent step in the output of net.

        Arguments:
            net (nn.Module or function): A backpropagatable function/module that receives
                images in (B x C x H x W) form and outputs a scalar value per image.
            src (torch.Tensor): Batch of images to update (B x C x H x W).
            step_size (float): Step size to use for the update: (im_old += step_size * grad)
            sigma (float): Standard deviation for gaussian smoothing (if used, see blur).
            precond (float): Strength of gradient smoothing.
            step_gain (float): Scaling factor for the step size.
            blur (boolean): Whether to blur the image after the update.
            jitter (int): Randomly shift the image this number of pixels before forwarding
                it through the network.
            eps (float): Small value to avoid division by zero.
            clip (boolean): Whether to clip the range of the image to be in [0, 255]
            train_norm (float): Decrease standard deviation of the image feed to the
                network to match this norm. Expressed in original pixel values. Unused if
                None
            norm (float): Decrease standard deviation of the image to match this norm after
                update. Expressed in z-scores. Unused if None
            add_loss (function): An additional term to add to the network activation before
                calling backward on it. Usually, some regularization.
        """
        if src.grad is not None:
            src.grad.zero_()

        # apply jitter shift
        if self.jitter > 0:
            ox, oy = np.random.randint(-self.jitter, self.jitter + 1, 2)  # use uniform distribution
            ox, oy = int(ox), int(oy)
            src.data = roll(roll(src.data, ox, -1), oy, -2)

        img = src
        if self.train_norm is not None and self.train_norm > 0.0:
            # normalize the image in backpropagatable manner
            img_idx = batch_std(src.data) + eps > self.train_norm / self.scale  # images to update
            if img_idx.any():
                img = src.clone()  # avoids overwriting original image but lets gradient through
                img[img_idx] = ((src[img_idx] / (batch_std(src[img_idx], keepdim=True) +
                                                 eps)) * (self.train_norm / self.scale))

        y = self.operation(img)
        (y.mean() + add_loss).backward()

        grad = src.grad
        if self.precond > 0:
            grad = fft_smooth(grad, self.precond)

        # src.data += (step_size / (batch_mean(torch.abs(grad.data), keepdim=True) + eps)) * (step_gain / 255) * grad.data
        a = step_size / (torch.abs(grad.data).mean() + eps)
        b = self.step_gain * grad.data  # itt (step gain -255) volt az egyik szorzó
        src.data += a * b
        # * both versions are equivalent for a single-image batch, for batches with more than
        # one image the first one is better but it drawns out the gradients that are spatially
        # wide; for instance a gradient of size 5 x 5 pixels all at amplitude 1 will produce a
        # higher change in an image of the batch than a gradient of size 20 x 20 all at
        # amplitude 1 in another. This is alright in most cases, but when generating diverse
        # images with min linkage (i.e, all images receive gradient from the signal and two
        # get the gradient from the diversity term) it drawns out the gradient generated from
        # the diversity term (because it is usually bigger spatially than the signal gradient)
        # and becomes hard to find very diverse images (i.e., increasing the diversity term
        # has no effect because the diversity gradient gets rescaled down to smaller values
        # than the signal gradient)
        # In any way, gradient mean is only used as normalization here and using the mean is
        # alright (also image generation works normally).

        # print(src.data.std() * scale)
        if self.norm is not None and self.norm > 0.0:
            data_idx = batch_std(src.data) + eps > self.norm / self.scale
            src.data[data_idx] = (src.data / (batch_std(src.data, keepdim=True) + eps) * self.norm / self.scale)[
                data_idx]

        if self.jitter > 0:
            # undo the shift
            src.data = roll(roll(src.data, -ox, -1), -oy, -2)

        if self.clip:
            src.data = torch.clamp(src.data, -self.bias / self.scale, (1 - self.bias) / self.scale)

        if self.blur:
            blur_in_place(src.data, sigma)


