import numpy as np
import scipy.ndimage
import scipy
import torch
from tqdm import tqdm
from ..tools.transforms import roll, batch_std
from ..tools.precond import fft_smooth

eps = 1e-12

def deepdraw(process, operation):
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

    for i in tqdm(range(process.iter_n)):
        # Diversity for multiple images (Image based)
        if process.diverse:
            div_term = diversion_term(process, process.div_metric,
                                      process.div_linkage, process.div_weight, process.div_mask)
        else:
            div_term = 0

        make_step(process, operation, step_i=i, add_loss=div_term)

        #if i % process.save_every == 0:
        #    image = process.get_image().data.clone().cpu().numpy()
        #    process.image_history.append(image)

    result_stats = get_result_stats(process, operation)
    return result_stats


def make_step(process, operation, step_i, add_loss=0):
    """
    Update meitorch in place making a gradient ascent step in the output of net.

    :param src: Image(s) to updata
    :param step_size: Step size to use for the update: (im_old += step_size * grad)
    :param sigma: Standard deviation for gaussian smoothing (if used, see blur).
    :param eps: Small value to avoid division by zero.
    :param add_loss: An additional term to add to the network activation before
                        calling backward on it. Usually, some regularization.
    """
    process.optimizer.zero_grad()
    inputs = process.get_samples()

    if process.scaler:
        scale = process.scaler(step_i)
        img_scale = inputs.detach().numpy()
        scipy.ndimage.zoom(img_scale, (1, 1, scale, scale))
        inputs.data = torch.from_numpy(img_scale)

    # apply jitter shift
    if process.jitter:
        ox, oy = np.random.randint(-process.jitter, process.jitter + 1, 2)  # use uniform distribution
        ox, oy = int(ox), int(oy)
        inputs.data = roll(roll(inputs.data, ox, -1), oy, -2)

    # normalize the image in backpropagatable manner
    if process.train_norm and process.train_norm > 0.0:
        img_idx = batch_std(inputs.data) + eps > process.train_norm / process.scale  # images to update
        if img_idx.any():
            img = inputs.clone()  # avoids overwriting original image but lets gradient through
            img[img_idx] = ((img[img_idx] / (batch_std(img[img_idx], keepdim=True) +
                                             eps)) * (process.train_norm / process.scale))

    outputs = operation(inputs)
    loss = outputs["loss"].mean()
    process.loss_history.append(loss.item())
    loss += add_loss
    loss.backward()

    if process.precond:
        for param in process.parameters():
            smooth_grad = fft_smooth(param.grad, process.precond(step_i))
            print(smooth_grad.shape)
            param.grad.data.copy_(smooth_grad)

    process.optimizer.step()
    #process.schedule.step()

    if process.norm and process.norm > 0.0:
        data_idx = batch_std(inputs.data) + eps > process.norm / process.scale
        inputs.data[data_idx] = (inputs.data / (batch_std(inputs.data, keepdim=True) + eps) * process.norm / process.scale)[
            data_idx]

    if process.jitter:
        # undo the shift
        inputs.data = roll(roll(inputs.data, -ox, -1), -oy, -2)

    if process.clip:
        inputs.data = torch.clamp(inputs.data, -process.bias / process.scale, (1 - process.bias) / process.scale)

    if process.blur:
        process.blur(inputs.data)


def get_result_stats(process, operation):
    with torch.no_grad():
        img = process.get_image()
        losses = operation(img)
        activation = losses["activation"].data.cpu().numpy()
    #cont, vals, lim_contrast = Analyze.contrast_tuning(operation, img.detach().cpu().numpy())
    result_dict = dict(
        activation=activation,
        kl=losses["kl"].data.cpu().numpy()
        #monotonic=bool(np.all(np.diff(vals) >= 0)),
        #max_activation=np.max(vals),
        #max_contrast=cont[np.argmax(vals)],
        #sat_contrast=np.max(cont),
        #img_mean=img.mean(),
        #lim_contrast=lim_contrast,
    )

    return result_dict

def diversion_term(process, div_metric='correlation', div_linkage='minimum', div_weight=0, div_mask=1):
    """
    Compute the diversity term for a set of images.

    :param div_metric: What metric to use when computing pairwise differences.
    :param div_linkage: How to agglomerate pairwise distances.
    :param div_weight: Weight given to the diversity term in the objective function.
    :param div_mask: Array (height x width) used to mask irrelevant parts of the
                        image before calculating diversity.
    """
    img = process.get_image()
    mask = torch.tensor(div_mask, dtype=torch.float32, device=process.device)
    div_term = 0
    if div_weight > 0:
        # Compute distance matrix
        images = (img * mask).view(len(img), -1)  # num_images x num_pixels
        if div_metric == 'correlation':
            # computations restricted to the mask
            means = (images.sum(dim=-1) / mask.sum()).view(len(images), 1, 1, 1)
            residuals = ((img - means) * torch.sqrt(mask)).view(len(img), -1)
            ssr = (((img - means) ** 2) * mask).sum(-1).sum(-1).sum(-1)
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
    return div_term