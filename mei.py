#import scipy
import torch
import numpy as np
from tqdm import tqdm
from utils import process, unprocess, roll, fft_smooth, batch_std, blur_in_place

class MEI:
    def __init__(self, model, dataset,  **MEIParams):
        # model parameters
        self.model = model
        self.img_shape, self.bias, self.scale = self.prepare_data(dataset)
        print('Working with images with mu={}, sigma={}'.format(self.bias, self.scale))

        # mei parameters
        self.iter_n = 1000          if 'iter_n' not in MEIParams else MEIParams['iter_n']
        self.start_sigma = 1.5      if 'start_sigma' not in MEIParams else MEIParams['start_sigma']
        self.end_sigma = 0.01       if 'end_sigma' not in MEIParams else MEIParams['end_sigma']
        self.start_step_size = 3.0  if 'start_step_size' not in MEIParams else MEIParams['start_step_size']
        self.end_step_size = 0.125  if 'end_step_size' not in MEIParams else MEIParams['end_step_size']
        self.precond = 0.0#0.1          if 'precond' not in MEIParams else MEIParams['precond']
        self.step_gain = 0.1        if 'step_gain' not in MEIParams else MEIParams['step_gain']
        self.jitter = 0             if 'jitter' not in MEIParams else MEIParams['jitter']
        self.blur = True            if 'blur' not in MEIParams else MEIParams['blur']
        self.norm = -1              if 'norm' not in MEIParams else MEIParams['norm']
        self.train_norm = -1        if 'train_norm' not in MEIParams else MEIParams['train_norm']

        # result parameters
        self.mei = None
        self.activation = None
        self.monotonic = None
        self.max_contrast = None
        self.max_activation = None
        self.sat_contrast = None
        self.img_mean = None
        self.lim_contrast = None



    def propogate(self, neuron_id):

        print(f'Working on neuron_id={neuron_id}')

        def adj_model(x):
            return self.model(x)[:, neuron_id]

        octaves = [
            {
                'iter_n': int(self.iter_n),
                'start_sigma': float(self.start_sigma),
                'end_sigma': float(self.end_sigma),
                'start_step_size': float(self.start_step_size),
                'end_step_size': float(self.end_step_size),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = self.img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = self.deepdraw(adj_model, gen_image, octaves, random_crop=False)

        #remove dimensions with size 1
        mei = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=self.bias, sigma=self.scale)[None, ...]).to('cpu')
            activation = adj_model(img).data.cpu().numpy()[0]
        cont, vals, lim_contrast = self.contrast_tuning(adj_model, mei, self.bias, self.scale)

        self.mei = mei
        self.activation = activation
        self.monotonic = bool(np.all(np.diff(vals) >= 0))
        self.max_activation = np.max(vals)
        self.max_contrast = cont[np.argmax(vals)]
        self.sat_contrast = np.max(cont)
        self.img_mean = mei.mean()
        self.lim_contrast = lim_contrast



    def deepdraw(self, net, base_img, octaves, random_crop=True, original_size=None, device='cpu'):
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
        # prepare base image
        image = process(base_img, mu=self.bias, sigma=self.scale)  # (3,224,224)

        # get input dimensions from net
        if original_size is None:
            print('getting image size:')
            c, w, h = image.shape[-3:]
        else:
            c, w, h = original_size

        print("starting drawing")

        src = torch.zeros(1, c, w, h, requires_grad=True, device=device)

        for e, o in enumerate(octaves):
            if 'scale' in o:
                pass
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

                self.make_step(net, src, sigma=sigma, step_size=step_size)

                if i % 10 == 0:
                    print('finished step %d in octave %d' % (i, e))

                # insert modified image back into original image (if necessary)
                image[:, ox:ox + w, oy:oy + h] = src.data[0].cpu().numpy()

        # returning the resulting image
        return unprocess(image, mu=self.bias, sigma=self.scale)


    def make_step(self, net, src,
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


        blur = bool(self.blur)
        jitter = int(self.jitter)
        precond = float(self.precond)
        step_gain = float(self.step_gain)
        norm = float(self.norm)
        clip = True
        train_norm = float(self.train_norm)

        # apply jitter shift
        if jitter > 0:
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)  # use uniform distribution
            ox, oy = int(ox), int(oy)
            src.data = roll(roll(src.data, ox, -1), oy, -2)

        img = src
        if train_norm is not None and train_norm > 0.0:
            # normalize the image in backpropagatable manner
            img_idx = batch_std(src.data) + eps > train_norm / self.scale  # images to update
            if img_idx.any():
                img = src.clone() # avoids overwriting original image but lets gradient through
                img[img_idx] = ((src[img_idx] / (batch_std(src[img_idx], keepdim=True) +
                                                 eps)) * (train_norm / self.scale))

        y = net(img)
        (y.mean() + add_loss).backward()

        grad = src.grad
        if precond > 0:
            grad = fft_smooth(grad, precond)

        # src.data += (step_size / (batch_mean(torch.abs(grad.data), keepdim=True) + eps)) * (step_gain / 255) * grad.data
        a = (step_size / (torch.abs(grad.data).mean() + eps))
        b = (step_gain / 255) * grad.data
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
        if norm is not None and norm > 0.0:
            data_idx = batch_std(src.data) + eps > norm / self.scale
            src.data[data_idx] =  (src.data / (batch_std(src.data, keepdim=True) + eps) * norm / self.scale)[data_idx]

        if jitter > 0:
            # undo the shift
            src.data = roll(roll(src.data, -ox, -1), -oy, -2)

        if clip:
            src.data = torch.clamp(src.data, -self.bias / self.scale, (255 - self.bias) / self.scale)

        if blur:
            blur_in_place(src.data, sigma)


    def prepare_data(self, trainset):
        """
        Given a key to uniquely identify a dataset and a readout key corresponding to a single component within the
        scan, returns information pertinent to generating MEIs

        Args:
            key: a key that can uniquely identify a single entry from StaticMultiDataset * DataConfig
            readout_key: a specific readout key

        Returns:
            trainset, img_shape, mu, mu_beh, mu_eye, s - where mu and s are mean and stdev of input images.
        """
        shape = list(trainset.dataset.data.shape)
        if (len(shape) == 3):
            shape = [1] + shape[1:]
        else:
            shape = shape[1:]
        print(shape)
        img_shape = shape
        mu = 0.0
        variance = 0.0
        total_samples = 0

        for inputs, _ in trainset:
            batch_size = inputs.size(0)
            total_samples += batch_size
            mu += torch.mean(inputs)
            variance += torch.var(inputs, unbiased=False)

        mu /= total_samples
        variance /= total_samples

        # Calculate the standard deviation
        s = torch.sqrt(variance)
        return img_shape, mu.numpy(), s.numpy()


    def contrast_tuning(self, model, img, bias, scale, min_contrast=0.01, n=1000, linear=True, use_max_lim=False):
        mu = img.mean()
        delta = img - img.mean()
        vmax = delta.max()
        vmin = delta.min()

        min_pdist = delta[delta > 0].min()
        min_ndist = (-delta[delta < 0]).min()

        max_lim_gain = max((255 - mu) / min_pdist, mu / min_ndist)

        base_contrast = img.std()

        lim_contrast = 255 / (vmax - vmin) * base_contrast # maximum possible reachable contrast without clipping
        min_gain = min_contrast / base_contrast
        max_gain = min((255 - mu) / vmax, -mu / vmin)

        def run(x):
            with torch.no_grad():
                img = torch.Tensor(process(x[..., None], mu=bias, sigma=scale)[None, ...])#.cuda()
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
            img = np.clip(img, 0, 255)
            c = img.std()
            v = run(img).data.cpu().numpy()[0]
            cont.append(c)
            vals.append(v)

        vals = np.array(vals)
        cont = np.array(cont)

        return cont, vals, lim_contrast