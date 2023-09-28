import torch
import numpy as np
from tqdm import tqdm
from itertools import product

from meitorch.tools.transforms import roll
from meitorch.tools.masks import mask_image


class Analyze:

    @staticmethod
    def best_match(process, operation, dataloader, mask=None, **MaskParams):
        """
        Find the image that maximizes the activation of the operation

        :param dataloader: The dataset containing the images
        :param mask: The mask to be applied to the image (optional)
        :param MaskParams: The parameters of the specific mask
        :return: The activation and the image that maximizes the activation
        """
        img_activations = []
        for image, label in tqdm(dataloader.dataset):
            image = mask_image(image, mask, process.bias, **MaskParams)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)
            if type(image) is not torch.Tensor:
                image = torch.tensor(image, dtype=torch.float32, requires_grad=True, device=process.device)
            else:
                image.requires_grad = True
            y = operation(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        pos = np.argmax(img_activations)
        return img_activations[pos], dataloader.dataset[pos][0].squeeze(0)

    @staticmethod
    def masked(process, mask, **MaskParams):
        """
        Apply a mask to the image

        :param mask: The mask to be applied (gaussian, meitorch, tight_mei)
        :param MaskParams: The parameters for the specific mask
        :return: The masked image
        """
        if process.image is None:
            return None
        return mask_image(process.image, mask, process.bias, **MaskParams)

    @staticmethod
    def masked_responses_process(process, operatiom=None, mask='gaussian', **MaskParams):
        """
        Apply a mask to the image and return the activation of the operation

        :param mask: The mask to be applied (gaussian, meitorch, tight_mei)
        :param MaskParams: The parameters for the specific mask
        :return: The activation and the masked image
        """
        if process.image is None:
            return None
        _, masked_img_activations, masked_images = \
            Analyze.masked_responses([process.image], operatiom, mask, process.bias, **MaskParams)
        return masked_img_activations[0], masked_images[0]


    @staticmethod
    def jittered_responses(process, operation, jitter_size):
        """
        Jitter the image and return the activation of the operation

        :param jitter_size: The size of the jitter,
                            the image will be shifted by -jitter_size, ..., 0, ..., jitter_size
        :return: The activation and the jittered images
        """
        if process.image is None:
            return None

        # jitter_size = 0 vagy 5
        shift = list(enumerate(range(-jitter_size, jitter_size + 1)))
        activations = np.empty((len(shift), len(shift)))

        jiterred_images = []

        with torch.no_grad():
            img = torch.Tensor(process.image).to(process.device)

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                jiterred_images.append(jittered_img)
                activations[iy, ix] = operation(jittered_img).data.cpu().numpy()

        return activations, jiterred_images

    @staticmethod
    def shifted_response(process, operation, x_shift, y_shift):
        """
        Shift the image and return the activation of the operation

        :param x_shift: Shift on the x-axis
        :param y_shift: Shift on the y-axis
        :return: The activation and the shifted image
        """

        if process.image is None:
            return None

        shifted_mei = np.roll(np.roll(process.image, x_shift, 1), y_shift, 0)

        with torch.no_grad():
            shifted_mei = torch.Tensor(shifted_mei).to(process.device)
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