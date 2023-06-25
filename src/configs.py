import numpy as np

"""
Gabor parameters for grid search

height:         int         # (px) image height
width:          int         # (px) image width
phases:         longblob    # (degree) angle at which to start the sinusoid
wavelengths:    longblob    # (px) wavelength of the sinusoid (1 / spatial frequency)
orientations:   longblob    # (degree) counterclockwise rotation to apply (0 is horizontal, 90 vertical)
sigmas:         longblob    # (px) sigma of the gaussian mask used
dys:            longblob    # (px/height) amount of translation in y (positive moves downwards)
dxs:            longblob    # (px/width) amount of translation in x (positive moves right)
"""
gabor_ranges = \
    dict(
            height=36,
            width=64,
            phases=[0, 90, 180, 270],
            wavelengths=[4, 7, 10, 15, 20],
            orientations=np.linspace(0, 180, 8, endpoint=False),
            sigmas=[2, 3, 5, 7, 9],
            dys=np.linspace(-0.3, 0.3, 7),
            dxs=np.linspace(-0.3, 0.3, 13)
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
    dict(
        height=36,
        width=64,
        lower_phase=0,
        upper_phase=360,
        lower_wavelenght=4,
        upper_wavelength=20,
        lower_orientation=0,
        upper_orientation=180,
        lower_sigma=2,
        upper_sigma=9,
        lower_dy=-0.35,
        upper_dy=0.35,
        lower_dx=-0.35,
        upper_dx=0.35
    )