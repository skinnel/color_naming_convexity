""" Script to run the convexity evaluation for optimal solutions obtained from the IB-color naming model."""

import numpy as np
from ib_color_naming.src import ib_naming_model
from empirical_tests.utils import get_quasi_concavity_measure
from empirical_tests.wcs_data_processing import wcs_data_pull, create_color_space_table

# # Generate cognitive source distributions
# # Need a better way than random to generate some of these
# cog_source = np.random.rand(500, 3)
# cog_source /= cog_source.sum(axis=1)[:, None]
#
# # Generate optimal color-naming systems
# model = ib_naming_model.load_model()
# fit_model = model.fit(cog_source)
# qW_M_fit = fit_model[-1]

# Generate optimal color-naming systems from the fit model
model = ib_naming_model.load_model()
optimal_naming_systems = model.qW_M
# Working under the assumption that the chip number corresponds to the array row index number

# Pull the data and generate coordinates for wcs chips
chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
    wcs_data_path = '/Users/Lindsay/Documents/school/CLMS/Thesis/data/WCS-Data-20110316',
    CIELAB_path = '/Users/Lindsay/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
)
cielab_map_df = create_color_space_table(cielab_df=cielab_df, chip_df=chip_df)
color_coords = cielab_map_df.sort_values(by='chip_id', axis=0, ascending=True)
chip_nums = color_coords['chip_id']
color_coords = color_coords[['L','A','B']].values

# Get accuracy, complexity and IB measures for each qW_M for each (cognitive source, color-naming system) pair
# Possibly average across all qW_M distributions in a single color-naming system

# Find which optimal naming systems have non-trivial number of color terms
ind_list = []
for i in range(len(optimal_naming_systems)):
    if optimal_naming_systems[i].shape[1] > 1:
        ind_list.append(i)

# Get quasi-concavity measure of each cognitive source distribution

# Get quasi-concavity measure of each color-naming system
ib_qc_agg = {}
ib_qc_each = {}
for i in ind_list:
    qw = optimal_naming_systems[i]
    aggregate_qc = 0
    qc_values = {}
    chip_ct_vec = []
    qc_vec = []
    for color_id in range(0, qw.shape[1]):

        # Get probabilities and coordinates
        x_probs = qw[:, color_id]
        x_coords = color_coords.copy()

        # Get the quasi-convexity measure of the distribution defined by the specified color term.
        color_qc = get_quasi_concavity_measure(x_coordinates=x_coords, x_probs=x_probs, mesh=None)
        qc_values[color_id] = color_qc

        # Determine how many chips assign the highest probability to this color term in this color naming system.
        max_probs = np.amax(qw[:,1:], axis=1)
        chip_ct = sum(x_probs>=max_probs)
        chip_ct_vec.append(chip_ct)
        qc_vec.append(color_qc)

    # Below is for a temp fix until the to do above is implemented
    chip_ct_vec = np.array(chip_ct_vec)
    qc_vec = np.array(qc_vec)
    chip_ct_vec = chip_ct_vec / chip_ct_vec.sum()
    aggregate_qc = np.sum(chip_ct_vec * qc_vec)

    ib_qc_agg[i] = aggregate_qc
    ib_qc_each[i] = qc_values

# Plot quasi-concavity of color-naming systems against...
    #... quasi-concavity of cognitive source distribution
    #... accuracy
    #... complexity
    #... IB measure

# Repeat the above for some non-optimal color-naming systems

