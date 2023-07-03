import numpy as np
import torch
from tqdm import tqdm
from scipy import optimize
from scipy import ndimage, signal

from . import configs as config
from .process import GaborProcess
from .utils import adjust_img_stats, mask_image
from .neuron_query import adj_model


class InputOptimizerBase:

    def __init__(self, models, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        self.models = models if models is not None else []
        self.bias = bias
        self.scale = scale
        self.device = device
        self.img_shape = shape

    def add_model(self, model):
        self.models.append(model)

    def remove_model(self, model):
        self.models.remove(model)

    @staticmethod
    def masked_responses(images, operation, mask='gaussian', bias=0, factor=1.0, device='cpu'):
        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(x[None, ...], dtype=torch.float32, requires_grad=False, device=device)
            y = operation(x).data.cpu().numpy()[0]
            return y


        original_img_activations = []
        masked_img_activations = []
        masked_images = []
        for image in tqdm(images):
            original_img_activations.append(evaluate_image(image))
            masked_image = mask_image(image, mask, bias, factor)
            masked_images.append(masked_image)
            masked_img_activations.append(evaluate_image(masked_image))

        return original_img_activations, masked_img_activations, masked_images


    # TODO: nem fix am
    @staticmethod
    def compute_spatial_frequency(img):
        from matplotlib import pyplot as plt
        # Compute the 2D Fourier Transform
        fft_img = np.fft.fft2(img)

        # Shift the zero-frequency component to the center of the spectrum
        fft_img_shifted = np.fft.fftshift(fft_img)

        # Compute the magnitude spectrum (absolute value)
        magnitude_spectrum = np.abs(fft_img_shifted)

        # Compute the spatial frequencies
        rows, cols = img.shape
        freq_rows = np.fft.fftfreq(rows)
        freq_cols = np.fft.fftfreq(cols)

        # Display the magnitude spectrum
        plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
        plt.colorbar()
        plt.show()

        return freq_cols, freq_rows, magnitude_spectrum


class Gabor(InputOptimizerBase):
    def __init__(self, models, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        super().__init__(models, shape, bias, scale, device)
        self.ranges = config.gabor_ranges
        self.limits = config.gabor_limits

    def set_limits(self, **limits):
        for key, value in limits.items():
            self.limits[key] = value

    def set_ranges(self, **ranges):
        for key, value in ranges.items():
            self.ranges[key] = value

    def best_gabor(self, neuron_query=None, gabor_loader=None):
        """
        Find the most exciting gabor for cells in the neuron query

        :param neuron_query:
        :param gabor_loader:
        :return:
        """
        # TODO: mean/contrast_params

        if gabor_loader is None:
            gabor_loader = self.create_gabor_loader(self.ranges)

        operation = adj_model(self.models, neuron_query)

        # Evaluate all gabors
        activations = []
        with torch.no_grad():
            for i, gabor in tqdm(enumerate(gabor_loader)):
                # norm = gabors
                norm = (gabor["image"] - self.bias) / self.scale
                img = torch.Tensor(norm[None, :]).to(self.device)
                print(img.shape)
                img_activations = operation(img).cpu().numpy()
                activations.append(img_activations)
        activations = np.concatenate(activations)  # num_gabors x num_cells

        # Check we got all gabors and all cells
        if len(activations) != len(gabor_loader.dataset):
            raise ValueError('Some gabor patches did not get processed')

        results = []
        for neuron_id, neuron_activations in enumerate(activations.T):
            # Select best gabor
            best_idx = np.argmax(neuron_activations)
            best_activation = neuron_activations[best_idx]
            (best_phase, best_wavelength, best_orientation, best_sigma, best_dy,
             best_dx) = gabor_loader[best_idx]['params']

            results.append(GaborProcess(
                operation=operation,
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
        return results

    def optimal_gabor(self, neuron_query=None, target_mean=None, target_contrast=None):
        """
        Find parameters that produce an optimal gabor for this unit

        best_gabor:         longblob    # best gabor image
        best_seed:          int         # random seed used to obtain the best gabor
        best_activation:    float       # activation at the best gabor image
        best_phase:         float       # (degree) angle at which to start the sinusoid
        best_wavelength:    float       # (px) wavelength of the sinusoid (1 / spatial frequency)
        best_orientation:   float       # (degree) counterclockwise rotation to apply (0 is horizontal, 90 vertical)
        best_sigma:         float       # (px) sigma of the gaussian mask used
        best_dy:            float       # (px/height) amount of translation in y (positive moves downwards)
        best_dx:            float       # (px/width) amount of translation in x (positive moves right)
        """
        operation = adj_model(self.models, neuron_query)

        def neg_model_activation(params):
            # Get params
            params = [np.clip(p, l, u) for p, (l, u) in zip(params, self.limits)]  # *
            phase, wavelength, orientation, sigma, dy, dx = params
            # * some local optimization methods in scipy.optimize receive parameter bounds
            # as arguments, however, empirically they seem to have lower performance than
            # those that do not (like Nelder-Mead which I use below). In general, gradient
            # based methods did worse than direct search ones.

            # Create gabor
            gabor = self.create_gabor(height=self.img_shape[-2], width=self.img_shape[-1],
                                      phase=phase, wavelength=wavelength, orientation=orientation,
                                      sigma=sigma, dy=dy, dx=dx,
                                      target_mean=target_mean, target_contrast=target_contrast)

            # Compute activation
            with torch.no_grad():
                norm = (gabor - self.bias) / self.scale
                img = torch.Tensor(norm[None, None, :, :]).to(self.device)
                activation = operation(img).item()

            return -activation

        # Find best parameters (simulated annealing -> local search)
        best_params = None
        best_activation = np.inf
        best_seed = None
        for seed in tqdm([1, 12, 123, 1234, 12345]):  # try 5 diff random seeds
            res = optimize.dual_annealing(neg_model_activation, bounds=self.limits, no_local_search=True, maxiter=300, seed=seed)
            res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')

            if res.fun < best_activation:
                best_activation = res.fun
                best_params = res.x
                best_seed = seed
        if best_params is None:
            raise ValueError('No solution found')

        best_params = [np.clip(p, l, u) for p, (l, u) in zip(best_params, self.limits)]

        # Create best gabor
        best_gabor = self.create_gabor(height=self.img_shape[-2], width=self.img_shape[-1],
                                       phase=best_params[0], wavelength=best_params[1],
                                       orientation=best_params[2], sigma=best_params[3],
                                       dy=best_params[4], dx=best_params[5])
        best_activation = -neg_model_activation(best_params)

        return GaborProcess(seed=best_seed,
                            operation=operation,
                            image=best_gabor,
                            bias=self.bias,
                            scale=self.scale,
                            device=self.device,
                            activation=best_activation,
                            phase=best_params[0],
                            wavelength=best_params[1],
                            orientation=best_params[2],
                            sigma=best_params[3],
                            dy=best_params[4],
                            dx=best_params[5])

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
        """ Create a gabor patch (sinusoidal + gaussian).
    
        Arguments:
            height (int): Height of the image in pixels.
            width (int): Width of the image in pixels.
            phase (float): Angle at which to start the sinusoid in degrees.
            wavelength (float): Wavelength of the sinusoid (1 / spatial frequency) in pixels.
            orientation (float): Counterclockwise rotation to apply (0 is horizontal) in
                degrees.
            sigma (float): Sigma of the gaussian mask used in pixels
            dy (float): Amount of translation in y (positive moves down) in pixels/height.
            dx (float): Amount of translation in x (positive moves right) in pixels/height.
    
        Returns:
            Array of height x width shape with the required gabor.
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

        # if lum is not None:
        # upscale the image
        # lum_gabor = ndimage.zoom(gabor, zoom=zoom_factor, mode='reflect')
        # key['gabor_mu'] = lum_gabor.mean()
        # key['gabor_contrast'] = lum_gabor.std()

        # invert gamma transformation into image space
        # gabor = np.clip(f_inv(lum_gabor), 0, 255)

        # small_gabor = cv2.resize(gabor, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)

        return gabor.astype(np.float32)

    @staticmethod
    def create_gabor_loader(param_ranges):
        """
        Grid search over gabor parameters then return a loader with these
        :return: data loader with gabor patches
        """
        import itertools

        gabors = []
        param_combinations = itertools.product(*list(param_ranges.values()))
        tqdm.write(f'Creating gabors')
        for params in tqdm(param_combinations):
            param_values = dict(zip(param_ranges.keys(), params))
            gabor = Gabor.create_gabor(**param_values)
            gabors.append(dict(image=gabor, params=param_values))

        return gabors
