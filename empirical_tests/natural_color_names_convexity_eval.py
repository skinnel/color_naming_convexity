""" Tests the quasi-convexity of natural color naming systems, as defined by the data from the WCS. """

# Imports
import pandas as pd
import numpy as np

from empirical_tests.wcs_data_processing import wcs_data_pull, create_color_space_table, lang_id_map, \
    create_lang_table, create_lang_encoder_dist
from empirical_tests.utils import  get_quasi_concavity_measure

# Pull the data
chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
    wcs_data_path = '/Users/Lindsay/Documents/school/CLMS/Thesis/data/WCS-Data-20110316',
    CIELAB_path = '/Users/Lindsay/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
)

# Get QC measure for each natural language in WCS
lang_list = list(lang_df['lang_name'].values)
wcs_lang_qc_agg = {}
wcs_lang_qc_each = {}
for lang in lang_list:

    # Set language id
    lang_id = lang_id_map(language=lang, lang_df=lang_df)

    # Process the data
    cielab_map_df = create_color_space_table(cielab_df=cielab_df, chip_df=chip_df)
    term_ct_df = create_lang_table(term_df=term_df, vocab_df=vocab_df, lang_id=lang_id)
    term_probs = create_lang_encoder_dist(term_ct_df=term_ct_df)

    # Get the QC measure for each color-term-affiliated distribution and aggregate
    aggregate_qc = 0
    qc_values = {}
    # Below are for a temp fix until the to do below is implemented
    chip_ct_vec = []
    qc_vec = []
    for color_id in range(1, term_probs.shape[1]):

        # Get the probability distribution and coordinates specific to the color term.
        x_all = term_probs[['chip_id', color_id]].copy(deep=True)
        x_all.rename(columns={color_id: 'prob'}, inplace=True)
        x_all = x_all.merge(right=cielab_map_df, how='inner', on='chip_id')
        x_probs = x_all['prob'].values
        x_coords = x_all[['L', 'A', 'B']].values

        # Get the quasi-convexity measure of the distribution defined by the specified color term.
        color_qc = get_quasi_concavity_measure(x_coordinates=x_coords, x_probs=x_probs, mesh=None)
        qc_values[color_id] = color_qc

        # Determine how many chips assign the highest probability to this color term in this color naming system.
        #TODO: Figure out a better way to deal with multiple highest prob color terms
        max_probs = np.amax(term_probs.iloc[:, 1:], axis=1)
        chip_ct = sum(x_probs >= max_probs)
        #wt = chip_ct/term_probs.shape[0]
        chip_ct_vec.append(chip_ct)
        qc_vec.append(color_qc)

        # Update aggregate quasi-convexity measure for the color naming system.
        #aggregate_qc += wt * color_qc

    # Below is for a temp fix until the to do above is implemented
    chip_ct_vec = np.array(chip_ct_vec)
    qc_vec = np.array(qc_vec)
    chip_ct_vec = chip_ct_vec/chip_ct_vec.sum()
    aggregate_qc = np.sum(chip_ct_vec*qc_vec)

    wcs_lang_qc_agg[lang] = aggregate_qc
    wcs_lang_qc_each[lang] = qc_values

# Do the above for every color term
# Also determine the weight for every color term (how many chips have that color term as their max / total number of chips)
    # If multiple (n) color terms are max, then get 1/n added to weight numerator
# sum over all color terms of (color term weigh * color term qc) gives color naming system qc measure

# Questions:
    #1. best way to deal with the issue causing the to do above?