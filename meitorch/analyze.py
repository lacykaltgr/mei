import torch
import numpy as np
from tqdm import tqdm
from itertools import product

from .tools.base_transforms import roll
from .tools.masks import mask_image


class Analyze:

    @staticmethod
    def best_match(mei_result, operation, dataloader, mask=None, **MaskParams):
        """
        Find the image that maximizes the activation of the operation

        :param mei_result: The result object containing the optimized visualization
        :param operation: The optimization operation
        :param dataloader: The dataset containing the images
        :param mask: The mask to be applied to the image (optional)
        :param MaskParams: The parameters of the specific mask
        :return: The activation and the image that maximizes the activation
        """
        img_activations = []
        for image, label in tqdm(dataloader.dataset):
            image = mask_image(image, mask, mei_result.bias, **MaskParams)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)
            if type(image) is not torch.Tensor:
                image = torch.tensor(image, dtype=torch.float32, requires_grad=True, device=mei_result.device)
            else:
                image.requires_grad = True
            y = operation(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        pos = np.argmax(img_activations)
        return img_activations[pos], dataloader.dataset[pos][0].squeeze(0)

    @staticmethod
    def masked(mei_result, mask, **MaskParams):
        """
        Apply a mask to the image

        :param mei_result: The result object containing the optimized visualization
        :param mask: The mask to be applied (gaussian, meitorch, tight_mei)
        :param MaskParams: The parameters for the specific mask
        :return: The masked image
        """
        if mei_result.image is None:
            return None
        return mask_image(mei_result.image, mask, mei_result.bias, **MaskParams)

    @staticmethod
    def masked_responses_process(mei_result, operation=None, mask='gaussian', **MaskParams):
        """
        Apply a mask to the image and return the activation of the operation

        :param mei_result: The result object containing the optimized visualization
        :param operation: The operation on which the activation will be measured
        :param mask: The mask to be applied (gaussian, meitorch, tight_mei)
        :param MaskParams: The parameters for the specific mask
        :return: The activation and the masked image
        """
        if mei_result.image is None:
            return None
        _, masked_img_activations, masked_images = \
            Analyze.masked_responses([mei_result.image], operation, mask, mei_result.bias, **MaskParams)
        return masked_img_activations[0], masked_images[0]

    @staticmethod
    def jittered_responses(mei_result, operation, jitter_size):
        """
        Jitter the image and return the activation of the operation

        :param mei_result: The result object containing the optimized visualization
        :param operation: The operation on which the activation will be measured
        :param jitter_size: The size of the jitter,
                            the image will be shifted by -jitter_size, ..., 0, ..., jitter_size
        :return: The activation and the jittered images
        """
        if mei_result.image is None:
            return None

        # jitter_size = 0 vagy 5
        shift = list(enumerate(range(-jitter_size, jitter_size + 1)))
        activations = np.empty((len(shift), len(shift)))

        jiterred_images = []

        with torch.no_grad():
            img = torch.Tensor(mei_result.image).to(mei_result.device)

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                jiterred_images.append(jittered_img)
                activations[iy, ix] = operation(jittered_img).data.cpu().numpy()

        return activations, jiterred_images

    @staticmethod
    def shifted_response(mei_result, operation, x_shift, y_shift):
        """
        Shift the image and return the activation of the operation

        :param mei_result: The result object containing the optimized visualization
        :param operation: The operation on which the activation will be measured
        :param x_shift: Shift on the x-axis
        :param y_shift: Shift on the y-axis
        :return: The activation and the shifted image
        """

        if mei_result.image is None:
            return None

        shifted_mei = np.roll(np.roll(mei_result.image, x_shift, 1), y_shift, 0)

        with torch.no_grad():
            shifted_mei = torch.Tensor(shifted_mei).to(mei_result.device)
            activations = operation(shifted_mei).data.cpu().numpy()

        return activations, shifted_mei

    @staticmethod
    def masked_responses(images, operation, mask='gaussian', bias=0, device='cpu', **MaskParams):
        """
        For comparision of activision between original and masked samples

        :param images: the images to be masked
        :param operation: the operation on which activisions will be measured
        :param mask: the type of the mask to be applied
        :param bias: background color
        :param device: device
        :param MaskParams: the parameters of specific masks (more on this in utils)
        :return: activation of the original image, activation of the masked image, the masked image
        """

        def evaluate_image(x):
            if len(x.shape) == 2:
                x = np.expand_dims(x, axis=0)
            x = torch.tensor(x, dtype=torch.float32, requires_grad=False, device=device)
            y = operation(x).data.cpu().numpy()
            return y

        original_img_activations = []
        masked_img_activations = []
        masked_images = []

        for image in tqdm(images):
            original_img_activations.append(evaluate_image(image))
            masked_image = mask_image(image, mask, bias, operation=operation, **MaskParams)
            masked_images.append(masked_image)
            masked_img_activations.append(evaluate_image(masked_image))

        return original_img_activations, masked_img_activations, masked_images

    @staticmethod
    def compute_spatial_frequency(img):
        """
        Computes and plots the spatial frequency of a given image

        :param img: image to compute sf on
        :return: frequency of each column, row in arrays, magnitude spectrum
        """
        from matplotlib import pyplot as plt

        # Compute the 2D Fourier Transform
        fft_img = np.fft.fft2(img)

        # Shift the zero-frequency component to the center of the spectrum
        fft_img_shifted = np.fft.fftshift(fft_img)

        # Compute the magnitude spectrum (absolute value)
        magnitude_spectrum = np.abs(fft_img_shifted)

        if len(img.shape) == 2:
            rows, cols = img.shape
            plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
        else:
            channels, rows, cols = img.shape
            for channel in range(channels):
                plt.imshow(np.log1p(magnitude_spectrum[channel]), cmap='gray')

        plt.colorbar()
        plt.show()

        freq_rows = np.fft.fftfreq(rows)
        freq_cols = np.fft.fftfreq(cols)

        return freq_cols, freq_rows, magnitude_spectrum

    @staticmethod
    def contrast_tuning(model, img, min_contrast=0.01, n=1000, linear=True, use_max_lim=False, device='cpu'):
        """
        Computes the contrast tuning curve for the given image and model.

        :param model: The model
        :param img: The image to compute the tuning curve for
        :param min_contrast: The minimum contrast to use
        :param n: The number of points to use
        :param linear: Whether to use linearly spaced points
        :param use_max_lim: Whether to use the maximum possible contrast without clipping
        :param device: The device to use
        :return: The contrast tuning curve
        """
        mu = img.mean()
        delta = img - img.mean()
        vmax = delta.max()
        vmin = delta.min()

        min_pdist = delta[delta > 0].min()
        min_ndist = (-delta[delta < 0]).min()

        max_lim_gain = max((1 - mu) / min_pdist, mu / min_ndist)

        base_contrast = img.std()

        lim_contrast = 1 / (vmax - vmin) * base_contrast  # maximum possible reachable contrast without clipping
        min_gain = min_contrast / base_contrast
        max_gain = min((1 - mu) / vmax, -mu / vmin)

        def run(x):
            with torch.no_grad():
                img = torch.Tensor(x).to(device)
                result = model(img)["objective"]
            return result

        target = max_lim_gain if use_max_lim else max_gain

        if linear:
            gains = np.linspace(min_gain, target, n)
        else:
            gains = np.logspace(np.log10(min_gain), np.log10(target), n)
        vals = []
        cont = []

        for g in tqdm(gains):
            img = delta * g + mu
            img = np.clip(img, 0, 1)
            c = img.std()
            v = run(img).data.cpu().numpy()
            cont.append(c)
            vals.append(v)

        vals = np.array(vals)
        cont = np.array(cont)

        return cont, vals, lim_contrast
