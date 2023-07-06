import numpy as np

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
gabor_ranges = \
    dict(
        height=[36],
        width=[64],
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
gabor_limits = \
    [
        (0, 360),  # phase
        (4, 20),  # wavelength
        (0, 180),  # orientation
        (2, 9),  # sigma
        (-0.35, 0.35),  # dy
        (-0.35, 0.35)  # dx
    ]
