import torch
import numpy as np
import scipy

from .gabor import Gabor
from .utils import contrast_tuning
from .process import MEIProcess


class MEI(Gabor):
    """
    Class for generating more complex optimized inputs
    """
    def __init__(self, models=None, operation=None, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        super().__init__(models, operation, shape, bias, scale, device)

    def generate(self, neuron_query=None, **MEIParams):
        """
        Generate most exciting inputs
        Uses deepdraw to optimize images
        :param neuron_query: The queried neurons of the output layer.
        :param MEIParams: Additional parameters for the optimization process.
        :return: Process(es) with MEI images
        """

        processes = []
        for op in self.get_operations(neuron_query):
            process = MEIProcess(op, bias=self.bias, scale=self.scale, device=self.device, **MEIParams)

            # generate initial random image
            background_color = np.float32([self.bias] * self.img_shape[-1])
            gen_image = np.random.normal(background_color, self.scale / 20, self.img_shape)
            gen_image = np.clip(gen_image, -1, 1)

            # generate class visualization via octavewise gradient ascent
            mei = MEI.deepdraw(process, gen_image, random_crop=False)

            with torch.no_grad():
                img = torch.Tensor(gen_image).to(self.device)
                activation = process.operation(img).data.cpu().numpy()

            cont, vals, lim_contrast = contrast_tuning(op, mei, device=self.device)

            process.image = mei
            process.neuron_query = neuron_query
            process.activation = activation
            process.monotonic = bool(np.all(np.diff(vals) >= 0))
            process.max_activation = np.max(vals)
            process.max_contrast = cont[np.argmax(vals)]
            process.sat_contrast = np.max(cont)
            process.img_mean = mei.mean()
            process.lim_contrast = lim_contrast
            processes.append(process)

        return processes if len(processes) > 1 else processes[0]

    def gradient_rf(self, neuron_query=None, **MEIParams):
        """
        Generate most exciting inputs based on the linear function of the gradients of the input
        Uses deepdraw to optimize images
        :param neuron_query: The queried neurons of the output layer.
        :param MEIParams: Additional parameters for the optimization process.
        :return: Process(es) with GradientRF images
        """

        processes = []
        for op in self.get_operations(neuron_query):

            X = torch.zeros(1, *self.img_shape, device=self.device, requires_grad=True)
            y = op(X)
            y.backward()
            point_rf = X.grad.data.cpu().numpy().squeeze()
            rf = X.grad.data

            def linear_model(x):
                return (x * rf).sum()

            process = MEIProcess(linear_model, bias=self.bias, scale=self.scale, device=self.device, **MEIParams)

            # generate initial random image
            background_color = np.float32([self.bias] * self.img_shape[-1])
            gen_image = np.random.normal(background_color, self.scale / 20, self.img_shape)
            gen_image = np.clip(gen_image, -1, 1)

            # generate class visualization via octavewise gradient ascent
            rf = MEI.deepdraw(process, gen_image, random_crop=False)

            with torch.no_grad():
                img = torch.Tensor(gen_image).to(self.device)
                activation = op(img).data.cpu().numpy()

            cont, vals, lim_contrast = contrast_tuning(op, rf, self.bias, self.scale)

            process.image = rf
            process.monotonic = bool(np.all(np.diff(vals) >= 0))
            process.max_activation = np.max(vals)
            process.max_contrast = cont[np.argmax(vals)]
            process.sat_contrast = np.max(cont)
            process.point_rf = point_rf
            process.activation = activation

            processes.append(process)
        return processes if len(processes) > 1 else processes[0]

    @staticmethod
    def deepdraw(process, image, random_crop=True, original_size=None):
        """
        Generate an image by iteratively optimizing activity of net.

        :param process: Process object with operation and other parameters.
        :param image: Initial image (h x w x c)
        :param random_crop: If image to optimize is bigger than networks input image,
        optimize random crops of the image each iteration.
        :param original_size: (channel, height, width) expected by the network. If
                            None, it uses base_img's.
        :return: The optimized image
        """
        # get input dimensions from net
        if original_size is None:
            c, w, h = image.shape[-3:]
        else:
            c, w, h = original_size

        src = torch.zeros(1, c, w, h, requires_grad=True, device=process.device)

        for e, o in enumerate(process.octaves):
            if 'scale' in o:
                # resize by o['scale'] if it exists
                image = scipy.ndimage.zoom(image, (1, o['scale'], o['scale']))
            _, imw, imh = image.shape
            for i in range(o['iter_n']):
                if imw > w:
                    if random_crop:
                        mid_x = (imw - w) / 2.
                        width_x = imw - w
                        ox = np.random.normal(mid_x, width_x * 0.3, 1)
                        ox = int(np.clip(ox, 0, imw - w))
                        mid_y = (imh - h) / 2.
                        width_y = imh - h
                        oy = np.random.normal(mid_y, width_y * 0.3, 1)
                        oy = int(np.clip(oy, 0, imh - h))
                        # insert the crop into meitorch.data[0]
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

                process.make_step(src, sigma=sigma, step_size=step_size)

                # insert modified image back into original image (if necessary)
                image[:, ox:ox + w, oy:oy + h] = src.data[0].cpu().numpy()

        return image

    @staticmethod
    def diverse_deepdraw(process, image, random_crop=True, original_size=None,
                         div_metric='correlation', div_linkage='minimum', div_weight=0, div_mask=1):
        """
        Similar to deepdraw() but including a diversity term among all images a la
        Cadena et al., 2018.

        :param process: Process object with operation and other parameters.
        :param image: Expects a 4-d array (num_images x height x width x channels).
        :param random_crop: If image to optimize is bigger than networks input image,
                            optimize random crops of the image each iteration.
        :param original_size: (channel, height, width) expected by the network. If
                            None, it uses base_img's.
        :param div_metric: What metric to use when computing pairwise differences.
        :param div_linkage: How to agglomerate pairwise distances.
        :param div_weight: Weight given to the diversity term in the objective function.
        :param div_mask: Array (height x width) used to mask irrelevant parts of the
                            image before calculating diversity.
        :return: Array of optimized images
        """

        if len(image) < 2:
            raise ValueError('You need to pass at least two initial images. Did you mean to '
                             'use deepdraw()?')

        # get input dimensions from net
        if original_size is None:
            c, w, h = image.shape[-3:]
        else:
            c, w, h = original_size

        src = torch.zeros(len(image), c, w, h, requires_grad=True, device=process.device)
        mask = torch.tensor(div_mask, dtype=torch.float32, device=process.device)

        for e, o in enumerate(process.octaves):
            if 'scale' in o:
                # resize by o['scale'] if it exists
                image = scipy.ndimage.zoom(image, (1, 1, o['scale'], o['scale']))
            imw, imh = image.shape[-2:]
            for i in range(o['iter_n']):
                if imw > w:
                    if random_crop:
                        mid_x = (imw - w) / 2.
                        width_x = imw - w
                        ox = np.random.normal(mid_x, width_x * 0.3, 1)
                        ox = int(np.clip(ox, 0, imw - w))
                        mid_y = (imh - h) / 2.
                        width_y = imh - h
                        oy = np.random.normal(mid_y, width_y * 0.3, 1)
                        oy = int(np.clip(oy, 0, imh - h))
                        # insert the crop into meitorch.data[0]
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

                process.make_step(src, sigma=sigma, step_size=step_size, add_loss=div_term)

                # insert modified image back into original image (if necessary)
                image[..., ox:ox + w, oy:oy + h] = src.detach().cpu().numpy()

        # returning the resulting image
        return image





