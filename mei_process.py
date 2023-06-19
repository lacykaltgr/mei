#import scipy
import torch
import numpy as np
from skimage.morphology import convex_hull_image, erosion, square
from scipy.ndimage.filters import gaussian_filter
from utils import fit_gauss_envelope, remove_small_area, adj_model
from tqdm import tqdm
#from scipy import signal, ndimage
#from scipy.ndimage import label


#TODO: add support for multiple octaves
class MEIProcess:
    def __init__(self, operation, bias=0, scale=1, device='cpu', **MEIParams):
        self.operation = operation
        self.device = device
        self.bias = bias
        self.scale = scale

        # result parameters
        self.mei = None
        self.activation = None
        self.monotonic = None
        self.max_contrast = None
        self.max_activation = None
        self.sat_contrast = None
        self.img_mean = None
        self.lim_contrast = None
        self.point_rf = None

        # mei parameters
        self.iter_n = 1000          if 'iter_n' not in MEIParams else MEIParams['iter_n']
        self.start_sigma = 1.5      if 'start_sigma' not in MEIParams else MEIParams['start_sigma']
        self.end_sigma = 0.01       if 'end_sigma' not in MEIParams else MEIParams['end_sigma']
        self.start_step_size = 3.0  if 'start_step_size' not in MEIParams else MEIParams['start_step_size']
        self.end_step_size = 0.125  if 'end_step_size' not in MEIParams else MEIParams['end_step_size']
        self.precond = 0#.1          if 'precond' not in MEIParams else MEIParams['precond']
        self.step_gain = 0.1        if 'step_gain' not in MEIParams else MEIParams['step_gain']
        self.jitter = 0             if 'jitter' not in MEIParams else MEIParams['jitter']
        self.blur = True            if 'blur' not in MEIParams else MEIParams['blur']
        self.norm = -1              if 'norm' not in MEIParams else MEIParams['norm']
        self.train_norm = -1        if 'train_norm' not in MEIParams else MEIParams['train_norm']
        self.clip = True            if 'clip' not in MEIParams else MEIParams['clip']

        self.octaves = [
            {
                'iter_n': self.iter_n,
                'start_sigma': self.start_sigma,
                'end_sigma': self.end_sigma,
                'start_step_size': self.start_step_size,
                'end_step_size': self.end_step_size,
            },
        ]

    def _gaussian_mask(self, factor):
        *_, mask = fit_gauss_envelope(self.mei)
        return mask ** factor

    def _gaussian_mask_with_info(self, factor):
        mu, cov, mask = fit_gauss_envelope(self.mei)
        mu_x = mu[0]
        mu_y = mu[1]
        cov_x = cov[0, 0]
        cov_y = cov[1, 1]
        cov_xy = cov[0, 1]
        mei_mask = mask ** factor
        return mei_mask, mu_x, mu_y, cov_x, cov_y, cov_xy

    def _mei_mask(self, delta_thr=16, size_thr=50, expansion_sigma=3, expansion_thr=0.3, filter_sigma=2):
        img = self.mei.copy(order='c')
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


    def _mei_tight_mask(self, stdev_size_thr=1, filter_sigma=1, target_reduction_ratio=0.9):
        # set in "c" contiguous
        img = self.mei.copy(order='c')

        def get_activation(mei):
            with torch.no_grad():
                img = torch.Tensor(mei[..., None]).to(self.device)
                activation = self.operation(img).data.cpu().numpy()[0]
            return activation

        delta = img - img.mean()
        fluc = np.abs(delta)
        thr = np.std(fluc) * stdev_size_thr

        # original mask
        mask = convex_hull_image((fluc > thr).astype(float))
        fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
        masked_img = fm * img + (1 - fm) * img.mean()
        activation = base_line = get_activation(masked_img)

        count = 0
        while (activation > base_line * target_reduction_ratio):
            mask = erosion(mask, square(3))
            fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
            masked_img = fm * img + (1 - fm) * img.mean()
            activation  = get_activation(masked_img)
            count += 1

            if count > 100:
                print('This has been going on for too long! - aborting')
                raise ValueError('The activation does not reduce for the given setting')

        reduction_ratio = activation / base_line
        return fm, reduction_ratio

    @staticmethod
    def mask_image(img, mask, background):
        """
        Applies the mask `mask` onto the `img`. The completely masked area is then
        replaced with the value `background`.


        Returns: masked image
        """
        filler = np.full_like(img, background)
        return img * mask + filler * (1-mask)



    def best_match(self, dataloader, mask=None, factor=1.0):

        if mask is None:
            def mask(img):
                return img
        elif mask == 'gaussian':
            def mask(img):
                return MEIProcess.mask_image(img, self._gaussian_mask(factor), self.bias)
        elif mask == 'mei':
            def mask(img):
                return MEIProcess.mask_image(img, self._mei_mask(), self.bias)
        elif mask == 'mei_tight':
            def mask(img):
                return MEIProcess.mask_image(img, self._mei_tight_mask(), self.bias)

        img_activations = []
        for image in tqdm(dataloader):
            image = np.atleast_3d(mask(image))  # ensure channel dimension exist
            image = torch.tensor(image[None, ...], dtype=torch.float32, requires_grad=True, device=self.device)
            # --- Compute gradient receptive field at the image
            y = self.operation(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        pos = np.argmax(img_activations)
        return img_activations[pos], dataloader.dataset[pos]




    #TODO: paraméterek még kellenének
    def responses(self, images, mask='gaussian', factor=1.0):
        if mask == 'gaussian':
            mask_fn = self._gaussian_mask(factor)
        elif mask == 'mei':
            mask_fn = self._mei_mask()
        elif mask == 'mei_tight':
            mask_fn = self._mei_tight_mask()
        else:
            raise NotImplementedError(f'Unknown mask type: {mask}')

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(x[None, ...], dtype=torch.float32, requires_grad=False, device='cuda')
            y = self.operation(x)
            return y.item()

        original_img_activations = []
        masked_img_activations = []
        for image in tqdm(images):
            original_img_activations.append(evaluate_image(image))
            masked_img_activations.append(evaluate_image(MEIProcess.mask_image(image, mask_fn, self.bias)))

        return original_img_activations, masked_img_activations


    #TODO: nem fix am
    def compute_spatial_frequency(self, img):
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