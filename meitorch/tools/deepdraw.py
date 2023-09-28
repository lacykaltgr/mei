import numpy as np
import scipy.ndimage
import scipy
import torch
from meitorch.tools.transforms import roll, batch_std, contrast_tuning
from meitorch.tools.denoisers import Gaussian
from meitorch.tools.precond import fft_smooth

def deepdraw(process, operation, iter_n=1000, start_sigma=1.5, end_sigma=0.01):
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
    image = process.image
    c, w, h = image.shape[-3:]
    src = torch.zeros(1, c, w, h, requires_grad=True, device=process.device)

    _, imw, imh = image.shape
    for i in range(iter_n):
        src.data[0].copy_(torch.Tensor(image))
        sigma = start_sigma + ((end_sigma - start_sigma) * i) / iter_n
        make_step(process, operation, sigma=sigma)
        if i % process.save_every == 0:
            image = process.get_image().copy().cpu().numpy()
            process.image_history.append(image)
    result_stats = process.get_result_stats(image, operation)
    return result_stats


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

def make_step(process, operation, sigma=None, eps=1e-12, add_loss=0):
    """
    Update meitorch in place making a gradient ascent step in the output of net.

    :param src: Image(s) to updata
    :param step_size: Step size to use for the update: (im_old += step_size * grad)
    :param sigma: Standard deviation for gaussian smoothing (if used, see blur).
    :param eps: Small value to avoid division by zero.
    :param add_loss: An additional term to add to the network activation before
                        calling backward on it. Usually, some regularization.
    """

    inputs = process.get_samples()

    if inputs.grad is not None:
        inputs.grad.zero_()

    # apply jitter shift
    if process.jitter > 0:
        ox, oy = np.random.randint(-process.jitter, process.jitter + 1, 2)  # use uniform distribution
        ox, oy = int(ox), int(oy)
        inputs.data = roll(roll(inputs.data, ox, -1), oy, -2)

    #img = src
    if process.train_norm is not None and process.train_norm > 0.0:
        # normalize the image in backpropagatable manner
        img_idx = batch_std(inputs.data) + eps > process.train_norm / process.scale  # images to update
        if img_idx.any():
            img = inputs.clone()  # avoids overwriting original image but lets gradient through
            img[img_idx] = ((img[img_idx] / (batch_std(img[img_idx], keepdim=True) +
                                             eps)) * (process.train_norm / process.scale))

    outputs = operation(inputs)
    loss = outputs.mean()
    process.loss_history.append(loss.item())
    (loss + add_loss).backward()

    #TODO: ez lehet az optimizerbe kéne???
    grad = inputs.grad
    if process.precond > 0:
        grad = fft_smooth(grad, process.precond)

    process.optimizer.step()

    if process.norm is not None and process.norm > 0.0:
        data_idx = batch_std(inputs.data) + eps > process.norm / process.scale
        inputs.data[data_idx] = (inputs.data / (batch_std(inputs.data, keepdim=True) + eps) * process.norm / process.scale)[
            data_idx]

    if process.jitter > 0:
        # undo the shift
        inputs.data = roll(roll(inputs.data, -ox, -1), -oy, -2)

    if process.clip:
        inputs.data = torch.clamp(inputs.data, -process.bias / process.scale, (1 - process.bias) / process.scale)

    if process.blur:
        Gaussian.blur_in_place(inputs.data, sigma)


def get_result_stats(self, process, operation):
    with torch.no_grad():
        img = process.get_image()
        activation = operation(img).data.cpu().numpy()
    cont, vals, lim_contrast = contrast_tuning(operation, img, device=self.device)
    result_dict = dict(
        activation=activation,
        monotonic=bool(np.all(np.diff(vals) >= 0)),
        max_activation=np.max(vals),
        max_contrast=cont[np.argmax(vals)],
        sat_contrast=np.max(cont),
        img_mean=img.mean(),
        lim_contrast=lim_contrast,
    )
    return result_dict
