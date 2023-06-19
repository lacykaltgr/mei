is_cuda = lambda m: next(m.parameters()).is_cuda
import torch
import numpy as np
#from scipy import ndimage
from itertools import product, zip_longest, count

def fft_smooth(grad, factor=1/4):
    """
    Tones down the gradient with 1/sqrt(f) filter in the Fourier domain.
    Equivalent to low-pass filtering in the spatial domain.
    """
    if factor == 0:
        return grad
    #h, w = grad.size()[-2:]
    # grad = tf.transpose(grad, [0, 3, 1, 2])
    # grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    h, w = grad.size()[-2:]
    # grad = tf.transpose(grad, [0, 3, 1, 2])
    # grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    tw = np.minimum(np.arange(0, w), np.arange(w-1, -1, -1), dtype=np.float32)  # [-(w+2)//2:]
    th = np.minimum(np.arange(0, h), np.arange(h-1, -1, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** factor)
    F = grad.new_tensor(t / t.mean()).unsqueeze(-1)
    print(F.shape)
    pp = torch.fft.rfft(grad.data, 2)
    print(pp.shape)
    return torch.fft.irfft(pp * F, 2)


def blur(img, sigma):
    if sigma > 0:
        for d in range(len(img)):
            pass
            #img[d] = ndimage.filters.gaussian_filter(img[d], sigma, order=0)
    return img


def blur_in_place(tensor, sigma):
    blurred = np.stack([blur(im, sigma) for im in tensor.cpu().numpy()])
    tensor.copy_(torch.Tensor(blurred))


def named_forward(self, input, name=None):
    for mod_name, module in self._modules.items():
        input = module(input)
        if mod_name == name:
            return input
    return input




def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def storeoutput(self, input, output):
    print(self.__class__.__name__)
    self._output = output
    raise Exception('Truncation')


def batch_mean(batch, keepdim=False):
    """ Compute mean for a batch of images. """
    mean = batch.view(len(batch), -1).mean(-1)
    if keepdim:
        mean = mean.view(len(batch), 1, 1, 1)
    return mean


def batch_std(batch, keepdim=False, unbiased=True):
    """ Compute std for a batch of images. """
    std = batch.view(len(batch), -1).std(-1, unbiased=unbiased)
    if keepdim:
        std = std.view(len(batch), 1, 1, 1)
    return std




'''
def diverse_deepdraw(net, base_img, octaves, random_crop=True, original_size=None,
                     bias=None, scale=None, device='cuda', div_metric='correlation',
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
    if len(base_img) < 2:
        raise ValueError('You need to pass at least two initial images. Did you mean to '
                         'use deepdraw()?')

    # prepare base image
    image = process(base_img, mu=bias, sigma=scale)  # (num_batches, num_channels, h, w)

    # get input dimensions from net
    if original_size is None:
        print('getting image size:')
        c, w, h = image.shape[-3:]
    else:
        c, w, h = original_size

    print("starting drawing")

    src = torch.zeros(len(image), c, w, h, requires_grad=True, device=device)
    mask = torch.tensor(div_mask, dtype=torch.float32, device=device)

    for e, o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = ndimage.zoom(image, (1, 1, o['scale'], o['scale']))
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

            make_step(net, src, bias=bias, scale=scale, sigma=sigma, step_size=step_size,
                      add_loss=div_term, **step_params)

            # TODO: Maybe save the MEIs every number of iterations and return all MEIs.
            if i % 10 == 0:
                print('finished step %d in octave %d' % (i, e))

            # insert modified image back into original image (if necessary)
            image[..., ox:ox + w, oy:oy + h] = src.detach().cpu().numpy()

    # returning the resulting image
    return unprocess(image, mu=bias, sigma=scale)


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

    return gabor.astype(np.float32)
'''





"""
def plot_images(df, prefixes, names=None, brain_area='V1', n_rows=15, order_by='pearson',
                panels=('normed_rf', 'normed_mei'), panel_names=('RF', 'MEI'), cmaps=('coolwarm', 'gray'),
                y_infos=('{prefix}test_corr', 'pearson'), save_path=None):
    if names is None:
        names = prefixes

    f = (df['brain_area'] == brain_area)
    area_data = df[f]
    area_data = area_data.sort_values(order_by, ascending=False)

    n_rows = min(n_rows, len(area_data))
    n_panels = len(panels)
    cols = len(prefixes) * n_panels;

    with sns.axes_style('white'):
        fig, axs = plt.subplots(n_rows, cols, figsize=(4 * cols, round(2 * n_cells)))
        st = fig.suptitle('MEIs on Shuffled {} dataset: {}'.format(brain_area, ', '.join(names)))
        [ax.set_xticks([]) for ax in axs.ravel()]
        [ax.set_yticks([]) for ax in axs.ravel()]

    for ax_row, (_, data_row), row_index in zip(axs, area_data.iterrows(), count()):
        for ax_group, prefix, name in zip(grouper(n_panels, ax_row), prefixes, names):
            for ax, panel, panel_name, y_info, cm in zip(ax_group, panels, panel_names, y_infos, cmaps):
                if row_index == 0:
                    ax.set_title('{}: {}'.format(panel_name, name))
                ax.imshow(data_row[prefix + panel].squeeze(), cmap=cm)
                if y_info is not None:
                    ax.set_ylabel('{:0.2f}%'.format(data_row[y_info.format(prefix=prefix)] * 100))

    fig.tight_layout()

    # shift subplots down:
    st.set_y(0.98)
    st.set_fontsize(20)
    fig.subplots_adjust(top=0.95)
    if path is not None:
        fig.savefig(save_path)


def gen_gif(images, output_path, duration=5, scale=1, adj_single=False):
    h, w = images[0].shape
    imgsize = (w * scale, h * scale)
    images = np.stack([cv2.resize(img, imgsize) for img in images])

    axis = (1, 2) if adj_single else None
    images = images - images.min(axis=axis, keepdims=True)
    images = images / images.max(axis=axis, keepdims=True) * 255
    images = images.astype('uint8')

    single_duration = duration / len(images)
    if not output_path.endswith('.gif'):
        output_path += '.gif'
    imageio.mimsave(output_path, images, duration=single_duration)
"""



def rescale_images(images, low=0, high=1, together=True):
    axis = None if together else (1, 2)
    images = images - images.min(axis=axis, keepdims=True)
    images = images / images.max(axis=axis, keepdims=True) * (high - low) + low
    return images

"""
def scale_imagesize(images, scale=(2, 2)):
    h, w = images[0].shape
    imgsize = (w * scale[1], h * scale[0])
    return np.stack([cv2.resize(img, imgsize) for img in images])
"""



def tile_images(images, rows, cols, vpad=0, hpad=0, normalize=False, base=0):
    n_images = len(images)
    assert rows * cols >= n_images
    h, w = images[0].shape

    total_image = np.zeros((h + (h + vpad) * (rows - 1), w + (w + hpad) * (cols - 1))) + base
    loc = product(range(rows), range(cols))
    for img, (i, j) in zip(images, loc):
        if normalize:
            img = rescale_images(img)
        voffset, hoffset = (h + vpad) * i, (w + hpad) * j
        total_image[voffset:voffset + h, hoffset:hoffset + w] = img
    return total_image



def repeat_frame(images, frame_pos=0, rep=4):
    parts = []
    if frame_pos < 0:
        frame_pos = len(images) + frame_pos

    if frame_pos > 0:
        parts.append(images[:frame_pos])
    parts.append(np.tile(images[frame_pos], (rep, 1, 1)))
    if frame_pos < len(images) - 1:
        parts.append(images[frame_pos+1:])
    return np.concatenate(parts)

"""
def add_text(image, text, pos, fontsize=1, color=(0, 0, 0)):
    image = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, text, pos, font, fontsize, color, 1, cv2.LINE_8)
    return image
"""


def query(x, query):
    for i in range(len(query)):
        x = x[:, query[i]]
    return x