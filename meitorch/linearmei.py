import numpy as np
import torch
from tqdm import tqdm
from scipy import optimize
from scipy import ndimage

from .objective.gabor import create_gabor_loader, create_gabor, default_gabor_ranges, default_gabor_limits
from .result import MEI_image


class LinearMEI:
    """
    Gabor filter creation, optimization, comparision
    Fewer features but a simpler interface then MEI
    """

    def __init__(self, operation, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        self.bias = bias
        self.scale = scale
        self.device = device
        self.img_shape = shape
        self.operation = operation
        self.ranges: dict = default_gabor_ranges
        self.limits: list = default_gabor_limits
        self.set_gabor_ranges(height=[self.img_shape[-2]], width=[self.img_shape[-1]])

    def best_gabor(self, gabor_loader=None):
        """
        Find the most exciting gabor filter for cells in the neuron query

        :param gabor_loader: The dataset containing the gabor filters to be evaluated
        :return: MEI_image for best gabor filter for the queried neurons
        """
        if self.img_shape[0] != 1:
            raise ValueError("Only grayscale images are supported for this feature, try using optimal gabor")

        if gabor_loader is None:
            gabor_loader = create_gabor_loader(self.ranges)

        # Evaluate all gabors
        activations = []
        with torch.no_grad():
            print('Evaluating gabors')
            for i, gabor in tqdm(enumerate(gabor_loader)):
                norm = (gabor["image"] - self.bias) / self.scale
                img = torch.Tensor(norm).to(self.device)
                activations.append(self.operation(img).cpu().numpy())
        activations = np.array(activations)

        best_idx = np.argmax(activations)
        (_, _, best_phase, best_wavelength, best_orientation, best_sigma, best_dy,
         best_dx) = gabor_loader[best_idx]['params'].values()
        return MEI_image(
            n_images=1,
            shape=self.img_shape,
            init=gabor_loader[best_idx]['image'],
            bias=self.bias,
            scale=self.scale,
            device=self.device,
            activation=activations[best_idx],
            phase=best_phase,
            wavelength=best_wavelength,
            orientation=best_orientation,
            sigma=best_sigma,
            dy=best_dy,
            dx=best_dx)

    def optimal_gabor(self, target_mean=None, target_contrast=None):
        """
        Find parameters that produce an optimal gabor for this queried neurons

        :param target_mean: Target mean of optimal gabor
        :param target_contrast: Target contrast of optimal gabor
        :return: MEI_image for optimal gabor filters for the queried neurons
        """

        bounds = self.limits * self.img_shape[0] if self.img_shape[0] > 1 else self.limits

        def neg_model_activation(params):
            params = [np.clip(p, l, u) for p, (l, u) in zip(params, bounds)]
            gabor = [create_gabor(height=self.img_shape[-2], width=self.img_shape[-1],
                                  phase=params[6 * channel], wavelength=params[6 * channel + 1],
                                  orientation=params[6 * channel + 2],
                                  sigma=params[6 * channel + 3],
                                  dy=params[6 * channel + 4], dx=params[6 * channel + 5],
                                  target_mean=target_mean, target_contrast=target_contrast)
                     for channel in range(self.img_shape[0])]
            gabor = np.stack(gabor, axis=0)

            with torch.no_grad():
                norm = (gabor - self.bias) / self.scale
                img = torch.Tensor(norm).to(self.device)
                activation = self.operation(img).item()

            return -activation

        best_params = None
        best_activation = np.inf

        # Find the best parameters (simulated annealing -> local search)
        for seed in tqdm([1, 12, 123, 1234, 12345]):  # try 5 diff random seeds
            res = optimize.dual_annealing(neg_model_activation, bounds=bounds, no_local_search=True,
                                          maxiter=300, seed=seed)
            res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')

            if res.fun < best_activation:
                best_activation = res.fun
                best_params = res.x
        if best_params is None:
            raise ValueError('No solution found')

        best_params = [np.clip(p, l, u) for p, (l, u) in zip(best_params, bounds)]

        # Create best gabor
        best_gabor = np.squeeze([create_gabor(height=self.img_shape[-2], width=self.img_shape[-1],
                                              phase=best_params[6 * i], wavelength=best_params[6 * i + 1],
                                              orientation=best_params[6 * i + 2], sigma=best_params[6 * i + 3],
                                              dy=best_params[6 * i + 4], dx=best_params[6 * i + 5])
                                 for i in range(self.img_shape[0])])
        best_activation = -neg_model_activation(best_params)

        return MEI_image(init=best_gabor,
                         shape=self.img_shape,
                         bias=self.bias,
                         scale=self.scale,
                         device=self.device,
                         activation=best_activation,
                         phase=[best_params[i] for i in range(self.img_shape[0])],
                         wavelength=[best_params[i + 1] for i in range(self.img_shape[0])],
                         orientation=[best_params[i + 2] for i in range(self.img_shape[0])],
                         sigma=[best_params[i + 3] for i in range(self.img_shape[0])],
                         dy=[best_params[i + 4] for i in range(self.img_shape[0])],
                         dx=[best_params[i + 5] for i in range(self.img_shape[0])])

    @staticmethod
    def white_noise_analysis(operation, shape, n_samples=1000, sigma=0.6, device="cpu"):
        if shape[0] == 1:
            shape = shape[1:]

        white_noise = np.random.normal(size=(n_samples, np.prod(shape)), loc=0.0, scale=1.)

        # apply ndimage.gaussian_filter with sigma=0.6
        for i in range(n_samples):
            white_noise[i, :] = ndimage.gaussian_filter(
                white_noise[i, :].reshape(shape), sigma=sigma).reshape(np.prod(shape))

        white_noise = torch.tensor(white_noise, dtype=torch.float32, device=device)
        values = []
        # loop over a batc h of 128 white_noise images
        with torch.no_grad():
            for i in range(0, n_samples, 128):
                batch = white_noise[i:i + 128, ...]
                batch = batch.reshape(-1, *shape)
                batch = torch.tensor(batch, dtype=torch.float32, device=device)
                values.append(operation(batch))
            values = torch.concatenate(values, dim=0)
        # multiply transpose of target block_values with white noise tensorially
        receptive_fields = torch.matmul(values.T, white_noise) / np.sqrt(n_samples)
        return receptive_fields

    def set_gabor_ranges(self, **ranges):
        """
        Ranges for the grid search for Gabor filter loader
        :param ranges: The ranges in key-value pairs
        """
        for key, value in ranges.items():
            self.ranges[key] = value

    def set_gabor_limits(self, **limits):
        """
        Bounds for Gabor filter optimization
        :param limits: The bounds in key-value pairs
        """

        def find_index(list_of_keys, element):
            for i, k in enumerate(list_of_keys):
                if k == element:
                    return i
            return -1

        for key, value in limits.items():
            index = find_index(self.ranges.keys(), key)
            if index == -1:
                raise ValueError("Invalid key")
            self.limits[index] = value
