""" Tests whether we can recreate the IB frontier from the Efficient Compression in Color Naming paper using the embo
package."""

import numpy as np
#import pandas as pd
#import pickle
from matplotlib import pyplot as plt
#from empirical_tests.wcs_data_processing import wcs_data_pull
#from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from embo import InformationBottleneck

# define joint probability mass function for a 2x2 joint pmf
pxy = np.array([[0.1, 0.4], [0.35, 0.15]])

# compute IB
I_x, I_y, H_m, beta = InformationBottleneck(pxy=pxy).get_bottleneck()

# plot I(M:Y) vs I(M:X)
plt.plot(I_x, I_y)
