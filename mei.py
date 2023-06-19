#import scipy
import torch
import numpy as np
from tqdm import tqdm
from utils import roll, fft_smooth, batch_std, blur_in_place, query
#from scipy import signal, ndimage


class MEI:
    def __init__(self, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        # model parameters
        self.device = device
        self.models = []
        self.bias = bias
        self.scale = scale
        self.img_shape = shape


    #TODO: add support for multiple octaves
    class MEIProcess:
        def __init__(self, operation, **MEIParams):
            self.operation = operation

            self.mei = None
            self.activation = None
            self.monotonic = None
            self.max_contrast = None
            self.max_activation = None
            self.sat_contrast = None
            self.img_mean = None
            self.lim_contrast = None

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

    def add_model(self, model):
        self.models.append(model)

    def remove_model(self, model):
        self.models.remove(model)

    def generate(self, neuron_query, **MEIParams):

        def adj_model(x):
            count = 0
            sum = None
            for model in self.models:
                y = query(model(x), neuron_query)
                sum = y if count == 0 else sum + y
                count += 1
            return sum / count

        process = self.MEIProcess(adj_model, **MEIParams)

        # generate initial random image
        background_color = np.float32([self.bias] * min(self.img_shape))
        gen_image = np.random.normal(background_color, self.scale / 20, self.img_shape)
        gen_image = np.clip(gen_image, -1, 1)

        # generate class visualization via octavewise gradient ascent
        gen_image = self.deepdraw(gen_image, process, random_crop=False)
        mei = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(gen_image[None, ...]).to(self.device)
            activation = adj_model(img).data.cpu().numpy()[0]

        #cont, vals, lim_contrast = MEI.contrast_tuning(adj_model, mei, device=self.device)

        process.mei = mei
        process.activation = activation
        #process.monotonic = bool(np.all(np.diff(vals) >= 0))
        #process.max_activation = np.max(vals)
        #process.max_contrast = cont[np.argmax(vals)]
        #process.sat_contrast = np.max(cont)
        #process.img_mean = mei.mean()
        #process.lim_contrast = lim_contrast

        return process


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

                self.make_step(process, src, sigma=sigma, step_size=step_size)

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

                self.make_step(process, src, sigma=sigma, step_size=step_size, add_loss=div_term)

                # TODO: Maybe save the MEIs every number of iterations and return all MEIs.
                if i % 10 == 0:
                    print('finished step %d in octave %d' % (i, e))

                # insert modified image back into original image (if necessary)
                image[..., ox:ox + w, oy:oy + h] = src.detach().cpu().numpy()

        # returning the resulting image
        return image


    def make_step(self, process, src,
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

    @staticmethod
    def contrast_tuning(model, img, min_contrast=0.01, n=1000, linear=True, use_max_lim=False, device='cpu'):
        mu = img.mean()
        delta = img - img.mean()
        vmax = delta.max()
        vmin = delta.min()

        min_pdist = delta[delta > 0].min()
        min_ndist = (-delta[delta < 0]).min()

        max_lim_gain = max((1 - mu) / min_pdist, mu / min_ndist)

        base_contrast = img.std()

        lim_contrast = 1 / (vmax - vmin) * base_contrast # maximum possible reachable contrast without clipping
        min_gain = min_contrast / base_contrast
        max_gain = min((1 - mu) / vmax, -mu / vmin)

        def run(x):
            with torch.no_grad():
                img = torch.Tensor(x[None, ...]).to(device)
                result = model(img)
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
            v = run(img).data.cpu().numpy()[0]
            cont.append(c)
            vals.append(v)

        vals = np.array(vals)
        cont = np.array(cont)

        return cont, vals, lim_contrast

    @staticmethod
    def create_gabor(height=36, width=64, phase=0, wavelength=10, orientation=0, sigma=5,
                     dy=0, dx=0):
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

        return gabor.astype(np.float32)
