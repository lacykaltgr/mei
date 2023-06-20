#import scipy
import torch
import numpy as np
from skimage.morphology import convex_hull_image, erosion, square
from scipy.ndimage.filters import gaussian_filter
from utils import fit_gauss_envelope, remove_small_area, adj_model, product, roll
from tqdm import tqdm
#from scipy import signal, ndimage
#from scipy.ndimage import label


#TODO: add support for multiple octaves
#TODO: lehessen specific operationnel (paraméterben megadva) használni
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

    def deepdraw(self, image, process, random_crop=True, original_size=None):
        """ Generate an image by iteratively optimizing activity of net.

        Arguments:
            net (nn.Module or function): A backpropagatable function/module that receives
                images in (B x C x H x W) form and outputs a scalar value per image.
            base_img (np.array): Initial image (h x w x c)
            octaves (list of dict): Configurations for each octave:
                n_iter (int): Number of iterations in this octave
                start_sigma (float): Initial standard deviation for gaussian smoothing (if
                    used, see blur)
                end_sigma (float): Final standard deviation for gaussian smoothing (if used,
                    see blur)
                start_step_size (float): Initial value of the step size used each iteration to
                    update the image (im_old += step_size * grad).
                end_step_size (float): Initial value of the step size used each iteration to
                    update the image (im_old += step_size * grad).
                (optionally) scale (float): If set, the image will be scaled using this factor
                    during optimization. (Original image size is left unchanged).
            random_crop (boolean): If image to optimize is bigger than networks input image,
                optimize random crops of the image each iteration.
            original_size (triplet): (channel, height, width) expected by the network. If
                None, it uses base_img's.
            bias (float), scale (float): Values used for image normalization (at the very
                start of processing): (base_img - bias) / scale.
            device (torch.device or str): Device where the network is located.
            step_params (dict): A handful of optional parameters that are directly sent to
                make_step() (see docstring of make_step for a description).

        Returns:
            A h x w array. The optimized image.
        """
        # get input dimensions from net
        if original_size is None:
            c, w, h = image.shape[-3:]
        else:
            c, w, h = original_size


        src = torch.zeros(1, c, w, h, requires_grad=True, device=self.device)

        for e, o in enumerate(process.octaves):
            if 'scale' in o:
                pass #TODO
                # resize by o['scale'] if it exists
                # image = scipy.ndimage.zoom(image, (1, o['scale'], o['scale']))
            _, imw, imh = image.shape
            for i in range(o['iter_n']):
                if imw > w:
                    if random_crop:
                        # randomly select a crop
                        # ox = random.randint(0,imw-224)
                        # oy = random.randint(0,imh-224)
                        mid_x = (imw - w) / 2.
                        width_x = imw - w
                        ox = np.random.normal(mid_x, width_x * 0.3, 1)
                        ox = int(np.clip(ox, 0, imw - w))
                        mid_y = (imh - h) / 2.
                        width_y = imh - h
                        oy = np.random.normal(mid_y, width_y * 0.3, 1)
                        oy = int(np.clip(oy, 0, imh - h))
                        # insert the crop into src.data[0]
                        src.data[0].copy_(torch.Tensor(image[:, ox:ox + w, oy:oy + h]))
                    else:
                        ox = int((imw - w) / 2)
                        oy = int((imh - h) / 2)
                        src.data[0].copy_(torch.Tensor(image[:, ox:ox + w, oy:oy + h]))
                else:
                    ox = 0
                    oy = 0
                    src.data[0].copy_(torch.Tensor(image))

                sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
                step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

                self._make_step(process, src, sigma=sigma, step_size=step_size)

                # insert modified image back into original image (if necessary)
                image[:, ox:ox + w, oy:oy + h] = src.data[0].cpu().numpy()

        return image


    def diverse_deepdraw(self, image, process, random_crop=True, original_size=None,
                         div_metric='correlation',
                         div_linkage='minimum', div_weight=0, div_mask=1, **step_params):
        """ Similar to deepdraw() but including a diversity term among all images a la
        Cadena et al., 2018.

        Arguments (only those additional to deepdraw):
            base_img: (CHANGED) Expects a 4-d array (num_images x height x width x channels).
            div_metric (str): What metric to use when computing pairwise differences.
            div_linkage (str): How to agglomerate pairwise distances.
            div_weight (float): Weight given to the diversity term in the objective function.
            div_mask (np.array): Array (height x width) used to mask irrelevant parts of the
                image before calculating diversity.
        """
        if len(image) < 2:
            raise ValueError('You need to pass at least two initial images. Did you mean to '
                             'use deepdraw()?')


        # get input dimensions from net
        if original_size is None:
            c, w, h = image.shape[-3:]
        else:
            c, w, h = original_size

        src = torch.zeros(len(image), c, w, h, requires_grad=True, device=self.device)
        mask = torch.tensor(div_mask, dtype=torch.float32, device=self.device)

        for e, o in enumerate(process.octaves):
            if 'scale' in o:
                pass #TODO
                # resize by o['scale'] if it exists
                #image = ndimage.zoom(image, (1, 1, o['scale'], o['scale']))
            imw, imh = image.shape[-2:]
            for i in range(o['iter_n']):
                if imw > w:
                    if random_crop:
                        # randomly select a crop
                        # ox = random.randint(0,imw-224)
                        # oy = random.randint(0,imh-224)
                        mid_x = (imw - w) / 2.
                        width_x = imw - w
                        ox = np.random.normal(mid_x, width_x * 0.3, 1)
                        ox = int(np.clip(ox, 0, imw - w))
                        mid_y = (imh - h) / 2.
                        width_y = imh - h
                        oy = np.random.normal(mid_y, width_y * 0.3, 1)
                        oy = int(np.clip(oy, 0, imh - h))
                        # insert the crop into src.data[0]
                        src.data[:].copy_(torch.Tensor(image[..., ox:ox + w, oy:oy + h]))
                    else:
                        ox = int((imw - w) / 2)
                        oy = int((imh - h) / 2)
                        src.data[:].copy_(torch.Tensor(image[..., ox:ox + w, oy:oy + h]))
                else:
                    ox = 0
                    oy = 0
                    src.data[:].copy_(torch.Tensor(image))

                sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
                step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

                div_term = 0
                if div_weight > 0:
                    # Compute distance matrix
                    images = (src * mask).view(len(src), -1)  # num_images x num_pixels
                    if div_metric == 'correlation':
                        # computations restricted to the mask
                        means = (images.sum(dim=-1) / mask.sum()).view(len(images), 1, 1, 1)
                        residuals = ((src - means) * torch.sqrt(mask)).view(len(src), -1)
                        ssr = (((src - means) ** 2) * mask).sum(-1).sum(-1).sum(-1)
                        distance_matrix = -(torch.mm(residuals, residuals.t()) /
                                            torch.sqrt(torch.ger(ssr, ssr)) + 1e-12)
                    elif div_metric == 'cosine':
                        image_norms = torch.norm(images, dim=-1)
                        distance_matrix = -(torch.mm(images, images.t()) /
                                            (torch.ger(image_norms, image_norms) + 1e-12))
                    elif div_metric == 'euclidean':
                        distance_matrix = torch.norm(images.unsqueeze(0) -
                                                     images.unsqueeze(1), dim=-1)
                    else:
                        raise ValueError('Invalid distance metric {} for the diversity term'.format(div_metric))

                    # Compute overall distance in this image set
                    triu_idx = torch.triu(torch.ones(len(distance_matrix),
                                                     len(distance_matrix)), diagonal=1) == 1
                    if div_linkage == 'minimum':
                        distance = distance_matrix[triu_idx].min()
                    elif div_linkage == 'average':
                        distance = distance_matrix[triu_idx].mean()
                    else:
                        raise ValueError('Invalid linkage for the diversity term: {}'.format(div_linkage))

                    div_term = div_weight * distance

                self._make_step(process, src, sigma=sigma, step_size=step_size, add_loss=div_term)

                # TODO: Maybe save the MEIs every number of iterations and return all MEIs.
                if i % 10 == 0:
                    print('finished step %d in octave %d' % (i, e))

                # insert modified image back into original image (if necessary)
                image[..., ox:ox + w, oy:oy + h] = src.detach().cpu().numpy()

        # returning the resulting image
        return image





    def _make_step(self, process, src,
                   step_size=1.5, sigma=None,
                   eps=1e-12, add_loss=0):
        """ Update src in place making a gradient ascent step in the output of net.

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
        if process.jitter > 0:
            ox, oy = np.random.randint(-process.jitter, process.jitter + 1, 2)  # use uniform distribution
            ox, oy = int(ox), int(oy)
            src.data = roll(roll(src.data, ox, -1), oy, -2)

        img = src
        if process.train_norm is not None and process.train_norm > 0.0:
            # normalize the image in backpropagatable manner
            img_idx = batch_std(src.data) + eps > process.train_norm / self.scale  # images to update
            if img_idx.any():
                img = src.clone() # avoids overwriting original image but lets gradient through
                img[img_idx] = ((src[img_idx] / (batch_std(src[img_idx], keepdim=True) +
                                                 eps)) * (process.train_norm / self.scale))

        y = process.operation(img)
        (y.mean() + add_loss).backward()

        grad = src.grad
        if process.precond > 0:
            grad = fft_smooth(grad, process.precond)

        # src.data += (step_size / (batch_mean(torch.abs(grad.data), keepdim=True) + eps)) * (step_gain / 255) * grad.data
        a = step_size / (torch.abs(grad.data).mean() + eps)
        b = process.step_gain * grad.data  #itt (step gain -255) volt az egyik szorzó
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

        #print(src.data.std() * scale)
        if process.norm is not None and process.norm > 0.0:
            data_idx = batch_std(src.data) + eps > process.norm / self.scale
            src.data[data_idx] =  (src.data / (batch_std(src.data, keepdim=True) + eps) * process.norm / self.scale)[data_idx]

        if process.jitter > 0:
            # undo the shift
            src.data = roll(roll(src.data, -ox, -1), -oy, -2)

        if process.clip:
            src.data = torch.clamp(src.data, -self.bias / self.scale, (1 - self.bias) / self.scale)

        if process.blur:
            blur_in_place(src.data, sigma)

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


    def jitter_in_place(self, jitter_size):

        #jitter_size = 0 vagy 5

        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(self.mei).to(self.device)

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                activations[iy, ix] = self.operation(jittered_img).data.cpu().numpy()[0]

        return activations

    def shift_in_place(self, x_shift, y_shift):

        shifted_mei = np.roll(np.roll(self.mei, x_shift, 1), y_shift, 0)

        with torch.no_grad():
            img = torch.Tensor(shifted_mei[..., None]).to(self.device)
            activations = self.operation(img).data.cpu().numpy()[0]

        return activations, shifted_mei
