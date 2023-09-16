import numpy as np
import torch
from tqdm import tqdm
from scipy import optimize
from scipy import ndimage, signal

from . import configs as config
from .process import GaborProcess
from .utils import adjust_img_stats, mask_image
from .neuron_query import adj_model


class _InputOptimizerBase:
    """
    Base class for operation based input optimization, mainly for future extendibility
    """

    def __init__(self, models=None, operation=None, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        self.models = models if models is not None else []
        self.bias = bias
        self.scale = scale
        self.device = device
        self.img_shape = shape
        self.operation = operation

    def add_model(self, model):
        self.models.append(model)

    def remove_model(self, model):
        self.models.remove(model)

    def get_operations(self, neuron_query=None):
        """
        :param neuron_query: specifies the query for the optimization (which neuron will be selected)
        :return: the operation on which the input will be optimized
        """
        if self.operation is not None:
            return [self.operation]
        elif len(self.models) > 0:
            return adj_model(self.models, neuron_query, input_shape=self.img_shape)
        else:
            raise ValueError("No valid operation")

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


class Gabor(_InputOptimizerBase):
    """
    Gabor filter creation, optimization, comparision
    Fewer features but a simpler interface then MEI
    """

    def __init__(self, models=None, operation=None, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        super().__init__(models, operation, shape, bias, scale, device)
        self.ranges: dict = config.gabor_ranges
        self.limits: list = config.gabor_limits
        self.set_ranges(height=[self.img_shape[-2]], width=[self.img_shape[-1]])

    def set_ranges(self, **ranges):
        """
        Ranges for the grid search for Gabor filter loader
        :param ranges: The ranges in key-value pairs
        """
        for key, value in ranges.items():
            self.ranges[key] = value

    def set_limits(self, **limits):
        """
        Bounds for Gabor filter optimization
        :param ranges: The bounds in key-value pairs
        """

        def find_index(list_of_keys, element):
            for index, key in enumerate(list_of_keys):
                if key == element:
                    return index
            return -1

        for key, value in limits.items():
            index = find_index(self.ranges.keys(), key)
            if index == -1:
                raise ValueError("Invalid key")
            self.limits[index] = value

    def best_gabor(self, neuron_query=None, gabor_loader=None):
        """
        Find the most exciting gabor filter for cells in the neuron query

        :param neuron_query: The queried neurons
        :param gabor_loader: The dataset containing the gabor filters to be evaluated
        :return: Process(es) for best gabor filter for the queried neurons
        """
        if self.img_shape[0] != 1:
            raise ValueError("Only grayscale images are supported for this feature, try using optimal gabor")

        if gabor_loader is None:
            gabor_loader = self.create_gabor_loader(self.ranges)

        operations = self.get_operations(neuron_query)

        # Evaluate all gabors
        activations = []
        with torch.no_grad():
            print('Evaluating gabors')
            for i, gabor in tqdm(enumerate(gabor_loader)):
                norm = (gabor["image"] - self.bias) / self.scale
                img = torch.Tensor(norm).to(self.device)
                img_activations = []
                for op in operations:
                    img_activations.append(op(img).cpu().numpy())
                img_activations = np.squeeze(img_activations)
                activations.append(img_activations)
        activations = np.array(activations)

        if activations.ndim == 1:
            best_idx = np.argmax(activations)
            (_, _, best_phase, best_wavelength, best_orientation, best_sigma, best_dy,
             best_dx) = gabor_loader[best_idx]['params'].values()
            return GaborProcess(
                image=gabor_loader[best_idx]['image'],
                operation=operations[0],
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

        activations = np.concatenate(activations)
        processes = []
        for neuron_id, neuron_activations in enumerate(activations.T):
            # Select best gabor
            best_idx = np.argmax(neuron_activations)
            best_activation = neuron_activations[best_idx]
            (_, _, best_phase, best_wavelength, best_orientation, best_sigma, best_dy,
             best_dx) = gabor_loader[best_idx]['params'].values()

            processes.append(GaborProcess(
                image=gabor_loader[best_idx]['image'],
                operation=operations[neuron_id],
                bias=self.bias,
                scale=self.scale,
                device=self.device,
                activation=best_activation,
                phase=best_phase,
                wavelength=best_wavelength,
                orientation=best_orientation,
                sigma=best_sigma,
                dy=best_dy,
                dx=best_dx))
        return processes

    def optimal_gabor(self, neuron_query=None, target_mean=None, target_contrast=None):
        """
        Find parameters that produce an optimal gabor for this queried neurons

        :param neuron_query: The neuron query
        :param target_mean: Target mean of optimal gabor
        :param target_contrast: Targer contrast of optimal gabor
        :return: Process(es) for optimal gabor filters for the queried neurons
        """
        operations = self.get_operations(neuron_query)
        processes = []
        bounds = self.limits * self.img_shape[0] if self.img_shape[0] > 1 else self.limits

        for op in operations:
            def neg_model_activation(params):
                params = [np.clip(p, l, u) for p, (l, u) in zip(params, bounds)]
                gabor = [self.create_gabor(height=self.img_shape[-2], width=self.img_shape[-1],
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
                    activation = op(img).item()

                return -activation

            best_params = None
            best_activation = np.inf
            best_seed = None

            # Find the best parameters (simulated annealing -> local search)
            for seed in tqdm([1, 12, 123, 1234, 12345]):  # try 5 diff random seeds
                res = optimize.dual_annealing(neg_model_activation, bounds=bounds, no_local_search=True,
                                              maxiter=300, seed=seed)
                res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')

                if res.fun < best_activation:
                    best_activation = res.fun
                    best_params = res.x
                    best_seed = seed
            if best_params is None:
                raise ValueError('No solution found')

            best_params = [np.clip(p, l, u) for p, (l, u) in zip(best_params, bounds)]

            # Create best gabor
            best_gabor = np.squeeze([self.create_gabor(height=self.img_shape[-2], width=self.img_shape[-1],
                                            phase=best_params[6 * i], wavelength=best_params[6 * i + 1],
                                            orientation=best_params[6 * i + 2], sigma=best_params[6 * i + 3],
                                            dy=best_params[6 * i + 4], dx=best_params[6 * i + 5])
                          for i in range(self.img_shape[0])])
            best_activation = -neg_model_activation(best_params)

            processes.append(GaborProcess(seed=best_seed,
                                          operation=op,
                                          image=best_gabor,
                                          bias=self.bias,
                                          scale=self.scale,
                                          device=self.device,
                                          activation=best_activation,
                                          phase=[best_params[i] for i in range(self.img_shape[0])],
                                          wavelength=[best_params[i + 1] for i in range(self.img_shape[0])],
                                          orientation=[best_params[i + 2] for i in range(self.img_shape[0])],
                                          sigma=[best_params[i + 3] for i in range(self.img_shape[0])],
                                          dy=[best_params[i + 4] for i in range(self.img_shape[0])],
                                          dx=[best_params[i + 5] for i in range(self.img_shape[0])]))
        return processes if len(processes) > 1 else processes[0]

    @staticmethod
    def create_gabor(
            height=36,
            width=64,
            phase=0,
            wavelength=10,
            orientation=0,
            sigma=5,
            dy=0,
            dx=0,
            target_mean=None,
            target_contrast=None,
            img_min=-1,
            img_max=1,
    ):
        """
        :param height: Height of the image in pixels.
        :param width: Width of the image in pixels.
        :param phase: Angle at which to start the sinusoid in degrees.
        :param wavelength: Wavelength of the sinusoid (1 / spatial frequency) in pixels.
        :param orientation: Counterclockwise rotation to apply (0 is horizontal) in
        degrees.
        :param sigma: Sigma of the gaussian mask used in pixels
        :param dy: Amount of translation in y (positive moves down) in pixels/height.
        :param dx: Amount of translation in x (positive moves right) in pixels/height.
        :param target_mean: Target mean of the image. If None, no normalization is applied.
        :param target_contrast: Target contrast of the image. If None, no normalization is applied.
        :param img_min: Minimum value of the image.
        :param img_max: Maximum value of the image.

        :return: Array of height x width shape with the required gabor.
        """
        # Compute image size to avoid translation or rotation producing black spaces
        padding = max(height, width)
        imheight = height + 2 * padding
        imwidth = width + 2 * padding
        # we could have diff pad sizes per dimension = max(dim_size, sqrt((h/2)^2 + (w/2)^2))
        # but this simplifies the code for just a bit of inefficiency

        # Create sinusoid with right wavelength and phase
        start_sample = phase
        step_size = 360 / wavelength
        samples = start_sample + step_size * np.arange(imheight)
        samples = np.mod(samples, 360)  # in degrees
        rad_samples = samples * (np.pi / 180)  # radians
        sin = np.sin(rad_samples)

        # Create Gabor by stacking the sinusoid along the cols
        gabor = np.tile(sin, (imwidth, 1)).T

        # Rotate around center
        gabor = ndimage.rotate(gabor, orientation, reshape=False)

        # Apply gaussian mask
        gaussy = signal.gaussian(imheight, std=sigma)
        gaussx = signal.gaussian(imwidth, std=sigma)
        mask = np.outer(gaussy, gaussx)
        gabor = gabor * mask

        # Translate (this is only approximate but it should be good enough)
        if abs(dx) > 1 or abs(dy) > 1:
            raise ValueError('Please express translations as factors of the height/width,'
                             'i.e, a number in interval [-1, 1] ')
        dy = int(dy * height)  # int is the approximation
        dx = int(dx * width)
        gabor = gabor[padding - dy: -padding - dy, padding - dx: -padding - dx]

        if gabor.shape != (height, width):
            raise ValueError('Dimensions of gabor do not match desired dimensions.')

        if target_mean is not None and target_contrast is not None:
            gabor, _ = adjust_img_stats(gabor, mu=target_mean, sigma=target_contrast, img_min=img_min, img_max=img_max)
        elif target_mean is not None or target_contrast is not None:
            raise ValueError('If you want to adjust the mean or contrast, you must specify both.')

        return gabor.astype(np.float32)

    @staticmethod
    def create_gabor_loader(param_ranges, load_path=None, save_path=None):
        """
        Grid search over gabor parameters then return a loader with these
        :param param_ranges: The ranges for the grid search
        :param load_path: Loading path the samples (new samples won't be created)
        :param save_path: The created samples will be saved here
        :return: Dataset with gabor filters
        """
        gabors = []

        if load_path is not None:
            import pickle
            with open(load_path, 'rb') as file:
                gabors = pickle.load(file)
            return gabors

        import itertools
        param_combinations = itertools.product(*list(param_ranges.values()))
        print(f'Creating gabors')
        for i, params in tqdm(enumerate(param_combinations)):
            param_values = dict(zip(param_ranges.keys(), params))
            gabor = Gabor.create_gabor(**param_values)
            gabors.append(dict(image=gabor, params=param_values))

        if save_path is not None:
            import pickle
            with open(save_path, 'wb') as file:
                pickle.dump(gabors, file)

        return gabors
