""" Module to generate the meaning distributions over the WCS data, as defined in Zaslavsky (2018)"""

# Imports
import numpy as np
from empirical_tests.wcs_data_processing import wcs_data_pull

def generate_WCS_meanings(perceptual_variance: float = 64) -> np.array:
    """
    A function to generate the meaning distributions over the observed colors in the World Color Survey (WCS).

    Arguments:
    ----------
    perceptual_variance
        The level of perceptual uncertainty between colors, as represented in LAB space. The default value, 64, is taken
        from Zaslavsky (2018).

    Returns:
    --------
    meanings
        An array of conditional distributions where each row corresponds to a meaning and the columns correspond to
        observations from the WCS. Distributions are conditioned by color chip.
    """

    # Pull the WCS data
    chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
        wcs_data_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/WCS-Data-20110316',
        CIELAB_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
    )

    # Generate meaning distributions, as defined in IB color paper
    u_vals = cielab_df['chip_id'].values
    meanings = np.zeros((len(u_vals), len(u_vals)))  # m x u
    # for mi in range(len(u_vals)):
    #     x1_val = cielab_df[['L', 'A', 'B']].iloc[mi].values
    #     for ui in range(len(u_vals)):
    #         if mi <= ui:
    #             x2_val = cielab_df[['L', 'A', 'B']].iloc[ui].values
    #             dist = np.linalg.norm(x1_val - x2_val)
    #             similarity = np.exp((-1 / (2 * perceptual_variance)) * (dist ** 2))
    #             meanings[mi, ui] = similarity
    #             if mi < ui:
    #                 meanings[ui, mi] = similarity
    for mi in range(len(u_vals)):
        m = int(cielab_df['chip_id'].iloc[mi]) - 1
        x1_val = cielab_df[['L', 'A', 'B']].iloc[m].values
        for ui in range(len(u_vals)):
            u = int(cielab_df['chip_id'].iloc[ui]) - 1
            if m <= u:
                x2_val = cielab_df[['L', 'A', 'B']].iloc[u].values
                dist = np.linalg.norm(x1_val - x2_val)
                similarity = np.exp((-1 / (2 * perceptual_variance)) * (dist ** 2))
                meanings[m, u] = similarity
                if m < u:
                    meanings[u, m] = similarity

    # Standardize meanings so they are true distributions
    for mi in range(len(u_vals)):
        meanings[mi, :] = meanings[mi, :] / np.sum(meanings[mi, :])
    #TODO: come back and parallelize this piece
    #meanings /= np.sum(meanings, axis=1)

    return meanings
