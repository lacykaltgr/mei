import numpy as np
from utils import adjust_img_stats, adj_model
import torch
from tqdm import tqdm


class Gabor:

    """
        height:         int         # (px) image height
    width:          int         # (px) image width
    phases:         longblob    # (degree) angle at which to start the sinusoid
    wavelengths:    longblob    # (px) wavelength of the sinusoid (1 / spatial frequency)
    orientations:   longblob    # (degree) counterclockwise rotation to apply (0 is horizontal, 90 vertical)
    sigmas:         longblob    # (px) sigma of the gaussian mask used
    dys:            longblob    # (px/height) amount of translation in y (positive moves downwards)
    dxs:            longblob    # (px/width) amount of translation in x (positive moves right)
    """
    ranges = [
        [1, 36, 64, [0, 90, 180, 270], [4, 7, 10, 15, 20], np.linspace(0, 180, 8, endpoint=False),
         [2, 3, 5, 7, 9], np.linspace(-0.3, 0.3, 7), np.linspace(-0.3, 0.3, 13)],
    ]


    """ # limits of some parameters search range to find the optimal gabor

    gaborlimits_id:     int         # id of this search range
    ---
    height:             int         # (px) height of image 
    width:              int         # (px) width of image
    lower_phase:        float
    upper_phase:       float
    lower_wavelength:   float
    upper_wavelength:   float
    lower_orientation:  float
    upper_orientation:  float
    lower_sigma:        float
    upper_sigma:        float
    lower_dy:           float
    upper_dy:           float
    lower_dx:           float
    upper_dx:           float
    """
    limits = [[1, 36, 64, 0, 360, 4, 20, 0, 180, 2, 9, -0.35, 0.35, -0.35, 0.35], ]

    def __init__(self, models, shape, bias=0, scale=1, device='cpu'):
        self.models = models
        self.bias = bias
        self.scale = scale
        self.device = device
        self.img_shape = shape

    def add_model(self, model):
        self.models.append(model)

    def remove_model(self, model):
        self.models.remove(model)

    def best_gabor(self, gabor_loader, neuron_query, **gabor_params):
        # find the most exciting gabor for each cell in this dataset
        #TODO: contrast_params

        readout_keys = None #TODO: neuron query

        for readout_key in readout_keys:
            # Evaluate all gabors
            activations = []
            with torch.no_grad():
                for i, gabors in tqdm(enumerate(gabor_loader)):
                    # norm = gabors
                    norm = (gabors - self.bias) / self.scale
                    img = torch.Tensor(norm[:, None, :, :]).to('cuda')
                    img_activations = self.operation(img).cpu().numpy()

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
                 best_dx) = dataset.args[best_idx][2:]


                # Insert
                results.append({'neuron_id': neuron_id,
                                'readout_key': readout_key,
                                'best_activation': best_activation,
                                'best_phase': best_phase,
                                'best_wavelength': best_wavelength,
                                'best_orientation': best_orientation,
                                'best_sigma': best_sigma, 'best_dy': best_dy,
                                'best_dx': best_dx})

    def optimal_gabor(
                self,
                target_mean=None,
                target_contrast=None,
        ):
        """ # find parameters that produce an optimal gabor for this unit

        -> TargetModel
        -> ProcessedImageConfig
        -> TargetDataset.Unit
        -> GaborLimits
        ---
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

        from scipy import optimize

        # Get optimization bounds per parameter
        bounds = self.limits

        # Write loss function to be optimized
        def neg_model_activation(params, bounds=bounds, height=self.limits['height'],
                                 width=self.limits['width'], target_mean=target_mean,
                                 target_contrast=target_contrast, train_mean=self.bias,
                                 train_std=self.scale, model=adj_model):
            # Get params
            params = [np.clip(p, l, u) for p, (l, u) in zip(params, bounds)]  # *
            phase, wavelength, orientation, sigma, dy, dx = params
            # * some local optimization methods in scipy.optimize receive parameter bounds
            # as arguments, however, empirically they seem to have lower performance than
            # those that do not (like Nelder-Mead which I use below). In general, gradient
            # based methods did worse than direct search ones.

            # Create gabor
            gabor = self.create_gabor(height=height, width=width, phase=phase,
                                      wavelength=wavelength, orientation=orientation,
                                      sigma=sigma, dy=dy, dx=dx, target_mean=target_mean,target_contrast=target_contrast)




            # Compute activation
            with torch.no_grad():
                norm = (gabor - train_mean) / train_std
                img = torch.Tensor(norm[None, None, :, :]).to('cuda')
                activation = model(img).item()

            return -activation

        # Find best parameters (simulated annealing -> local search)
        best_activation = np.inf
        for seed in tqdm([1, 12, 123, 1234, 12345]):  # try 5 diff random seeds
            res = optimize.dual_annealing(neg_model_activation, bounds=bounds,
                                          no_local_search=True, maxiter=300, seed=seed)
            res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')

            if res.fun < best_activation:
                best_activation = res.fun
                best_params = res.x
                best_seed = seed
        best_params = [np.clip(p, l, u) for p, (l, u) in zip(best_params, bounds)]

        # Create best gabor
        best_gabor = self.create_gabor(height=self.limits['height'], width=self.limits['width'],
                                       phase=best_params[0], wavelength=best_params[1],
                                       orientation=best_params[2], sigma=best_params[3],
                                       dy=best_params[4], dx=best_params[5])
        best_activation = -neg_model_activation(best_params)

        # Insert
        return {'best_gabor': best_gabor,
                'best_seed': best_seed,
                'best_activation': best_activation,
                'best_phase': best_params[0],
                'best_wavelength': best_params[1],
                'best_orientation': best_params[2],
                'best_sigma': best_params[3],
                'best_dy': best_params[4],
                'best_dx': best_params[5]}


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
        """ # lists of gabor parameters to search over for the best gabor

        gaborrange_id:  int     # id for each range
        ---
        height:         int         # (px) image height
        width:          int         # (px) image width
        phases:         longblob    # (degree) angle at which to start the sinusoid
        wavelengths:    longblob    # (px) wavelength of the sinusoid (1 / spatial frequency)
        orientations:   longblob    # (degree) counterclockwise rotation to apply (0 is horizontal, 90 vertical)
        sigmas:         longblob    # (px) sigma of the gaussian mask used
        dys:            longblob    # (px/height) amount of translation in y (positive moves downwards)
        dxs:            longblob    # (px/width) amount of translation in x (positive moves right)

        contents = [
            [1, 36, 64, [0, 90, 180, 270], [4, 7, 10, 15, 20], np.linspace(0, 180, 8, endpoint=False),
             [2, 3, 5, 7, 9], np.linspace(-0.3, 0.3, 7), np.linspace(-0.3, 0.3, 13)],
        ]
        """

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
        #gabor = ndimage.rotate(gabor, orientation, reshape=False)

        # Apply gaussian mask
        #gaussy = signal.gaussian(imheight, std=sigma)
        #gaussx = signal.gaussian(imwidth, std=sigma)
        #mask = np.outer(gaussy, gaussx)
        #gabor = gabor * mask

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

        #if lum is not None:
            # upscale the image
            #lum_gabor = ndimage.zoom(gabor, zoom=zoom_factor, mode='reflect')
            #key['gabor_mu'] = lum_gabor.mean()
            #key['gabor_contrast'] = lum_gabor.std()

            # invert gamma transformation into image space
            #gabor = np.clip(f_inv(lum_gabor), 0, 255)

            #small_gabor = cv2.resize(gabor, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)

        return gabor.astype(np.float32)


    @staticmethod
    def create_gabor_loader():
        pass
