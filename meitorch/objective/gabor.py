import numpy as np
from scipy import ndimage, signal
from tqdm import tqdm
from ..tools.base_transforms import adjust_img_stats


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
    """
    :param height: Height of the image in pixels.
    :param width: Width of the image in pixels.
    :param phase: Angle at which to start the sinusoid in degrees.
    :param wavelength: Wavelength of the sinusoid (1 / spatial frequency) in pixels.
    :param orientation: Counterclockwise rotation to apply (0 is horizontal) in
    degrees.
    :param sigma: Sigma of the gaussian mask used in pixels
    :param dy: Amount of translation in y (positive moves down) in pixels/height.
    :param dx: Amount of translation in x (positive moves right) in pixels/height.
    :param target_mean: Target mean of the image. If None, no normalization is applied.
    :param target_contrast: Target contrast of the image. If None, no normalization is applied.
    :param img_min: Minimum value of the image.
    :param img_max: Maximum value of the image.

    :return: Array of height x width shape with the required gabor.
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

    return gabor.astype(np.float32)


def create_gabor_loader(param_ranges, load_path=None, save_path=None):
    """
    Grid search over gabor parameters then return a loader with these
    :param param_ranges: The ranges for the grid search
    :param load_path: Loading path the samples (new samples won't be created)
    :param save_path: The created samples will be saved here
    :return: Dataset with gabor filters
    """
    gabors = []

    if load_path is not None:
        import pickle
        with open(load_path, 'rb') as file:
            gabors = pickle.load(file)
        return gabors

    import itertools
    param_combinations = itertools.product(*list(param_ranges.values()))
    print(f'Creating gabors')
    for i, params in tqdm(enumerate(param_combinations)):
        param_values = dict(zip(param_ranges.keys(), params))
        gabor = create_gabor(**param_values)
        gabors.append(dict(image=gabor, params=param_values))

    if save_path is not None:
        import pickle
        with open(save_path, 'wb') as file:
            pickle.dump(gabors, file)

    return gabors



"""
Gabor parameters for grid search

height:         int         # (px) image height
width:          int         # (px) image width
phases:         long        # (degree) angle at which to start the sinusoid
wavelengths:    long        # (px) wavelength of the sinusoid (1 / spatial frequency)
orientations:   long        # (degree) counterclockwise rotation to apply (0 is horizontal, 90 vertical)
sigmas:         long        # (px) sigma of the gaussian mask used
dys:            long        # (px/height) amount of translation in y (positive moves downwards)
dxs:            long        # (px/width) amount of translation in x (positive moves right)
"""
default_gabor_ranges = \
    dict(
        height=[32],
        width=[32],
        phase=[0, 90, 180, 270],
        wavelength=[4, 7, 10, 15, 20],
        orientation=np.linspace(0, 180, 8, endpoint=False),
        sigma=[2, 3, 5, 7, 9],
        dy=np.linspace(-0.3, 0.3, 7),
        dx=np.linspace(-0.3, 0.3, 13)
    )

""" 
Limits of some parameters search range to find the optimal Gabor

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
default_gabor_limits = \
    [
        (0, 360),  # phase
        (4, 20),  # wavelength
        (0, 180),  # orientation
        (2, 9),  # sigma
        (-0.35, 0.35),  # dy
        (-0.35, 0.35)  # dx
    ]