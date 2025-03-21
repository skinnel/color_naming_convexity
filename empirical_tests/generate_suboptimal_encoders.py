""" Generates a set of sub-optimal encoders from either the IB-optimale encoders or the natural languages, in order to
fill out the IB space below the frontier and get a better sense of how the QC value changes with respect to distance
from the frontier."""

# Imports
import pickle
import random
import numpy as np
import pandas as pd


from matplotlib import pyplot as plt
from typing import List, Dict, Tuple
from copy import deepcopy

from empirical_tests.wcs_data_processing import wcs_data_pull, create_color_space_table
from empirical_tests.utils import get_quasi_concavity_measure
from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from ib_from_scratch.ib_color_model_with_annealing import update_qw, update_mwu, drop_unused_dimensions
from ib_from_scratch.ib_utils import get_entropy, get_mutual_information


# Function to perform the encoder shuffling
def shuffle_encoder(enc_df: pd.DataFrame, pct_shuffle: float, r_state: int = None) -> pd.DataFrame:
    """ Shuffles the encoder rows (meanings axis) in order to create suboptimal encoders for experimentation.

    Arguments:
    ----------
    enc_df
        The original (optimal or NL) encoder.
    pct_shuffle
        A value between 0.0 and 1.0 indicating the fraction of the rows in enc_df that should be shuffled.

    Returns:
    --------
    new_enc_df
        The new, suboptimal encoder, with the shuffled rows.
    """

    # Shuffle the rows
    shuffled_rows = enc_df.sample(frac=pct_shuffle, replace=False, random_state=r_state, axis=0, ignore_index=False)
    shuffle_ind = list(shuffled_rows.index.copy(deep=True))
    shuffle_ind.sort()
    shuffled_rows['new_index'] = shuffle_ind
    shuffled_rows.set_index(keys='new_index', inplace=True)

    # Reintegrate the shuffled rows
    new_enc_df = enc_df.copy(deep=True)
    new_enc_df.loc[shuffled_rows.index, :] = shuffled_rows

    return new_enc_df


def generate_shuffled_encs(
        enc_dict: Dict[str, list], shuffle_pcts: List[float], enc_type: str, n_samples: int = 1,
        pm: pd.DataFrame = None, verbose: bool = False
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """ Generates a large collection of shuffled encoders using a set of user-provided encoders and list of shuffle
    percentages

    Arguments:
    ----------
    enc_dict
        The dictionary containing ordered lists of the affiliated beta, ib values and encoders.
    shuffle_pcts
        A list of the various percentage values to be used to generate the shuffled encoders
    enc_type
        Indicates if the shuffled encoders are being generated from synthetic encoders, which have beta and ib values,
        or from the natural languages, which have language ids and names.
    n_samples
        The number of times the shuffling should be performed for a single percent value.
    pm
        The probabilistic need distribution for the meanings.
    verbose
        If true logging statements are printed to standard output.

    Returns:
    --------
    shuffled_encs_dict
        Dictionary formatted like enc_dict that contains the newly shuffled encoders. Beta values and retained from the
        original encoder and IB values are filled in with a dummy 0.0.
    """

    # Initialize new shuffled encoder dictionary
    if enc_type == 'nl':
        shuffled_encs_dict = {
            'id': [],
            'lang_id': [],
            'pct_shuffled': [],
            'sample_number': [],
            'lang_name': [],
            'encoder': []
        }

        ib_res = {
            'id': [],
            'complexity': [],
            'accuracy': [],
            'entropy': [],
            'lang_name': [],
            'lang_id': [],
            'pct_shuffled': [],
            'sample_number': []
        }
    else:
        shuffled_encs_dict = {
            'id': [],
            'beta': [],
            'pct_shuffled': [],
            'sample_number': [],
            'ib_val': [],
            'encoder': []
        }

        ib_res = {
            'id': [],
            'complexity': [],
            'accuracy': [],
            'entropy': [],
            'ib': [],
            'beta': [],
            'pct_shuffled': [],
            'sample_number': []
        }

    # Get necessary pieces to generate the ib results
    meanings = generate_WCS_meanings(perceptual_variance=64)
    if pm is None:
        pm = np.ones(330) / 330
    pu_m = meanings

    # Get p(u) from p(m) and m(u) = p(u|m)
    # TODO: check columns vs rows in for loop
    pu = np.zeros(330)
    for mi in range(pu_m.shape[0]):
        mu = pu_m[mi, :]
        wt_m = pm[mi]
        pu_delta = wt_m * mu
        pu = pu + pu_delta

    # Get shuffled encoders
    if enc_type == 'nl':
        id_vec = enc_dict['lang_id']
    else:
        id_vec = enc_dict['beta']
    for i in range(len(id_vec)):
        if verbose:
            print(f'Creating new encoders for encoder {i}/{len(id_vec)}')
        if enc_type == 'nl':
            id = enc_dict['lang_id'][i]
            lang_name = enc_dict['lang_name'][i]
        else:
            id = enc_dict['beta'][i]
        enc = pd.DataFrame(enc_dict['encoder'][i])
        enc = enc.transpose()
        for pct in shuffle_pcts:
            for n in range(n_samples):

                # Define unique identifier
                id_val = f'{id}_{pct}_{n}'

                # Get new encoder
                new_enc = shuffle_encoder(enc, pct)
                new_enc = new_enc.copy(deep=True)
                new_enc = new_enc.transpose()
                new_enc = new_enc.values

                # Save new encoder
                shuffled_encs_dict['id'].append(id_val)
                shuffled_encs_dict['pct_shuffled'].append(pct)
                shuffled_encs_dict['sample_number'].append(n)
                shuffled_encs_dict['encoder'].append(new_enc)
                if enc_type == 'nl':
                    shuffled_encs_dict['lang_id'].append(id)
                    shuffled_encs_dict['lang_name'].append(lang_name)
                else:
                    shuffled_encs_dict['beta'].append(id)
                    shuffled_encs_dict['ib_val'].append(0.0)

                # Get accuracy, complexity and IB values for new encoder
                qw_df = update_qw(pm, new_enc)
                mwu_df = update_mwu(qw_df, pm, pu_m, new_enc)
                qw_df, new_enc_df, mwu_df = drop_unused_dimensions(qw_df, new_enc, mwu_df, eps=0.001)
                acc_val = get_mutual_information(pu, qw_df, mwu_df)
                comp_val = get_mutual_information(qw_df, pm, new_enc_df)
                ent_val = get_entropy(qw_df)
                if enc_type != 'nl':
                    ib_val = comp_val - (id * acc_val)

                # Save IB results
                ib_res['id'].append(id_val)
                ib_res['complexity'].append(comp_val)
                ib_res['accuracy'].append(acc_val)
                ib_res['entropy'].append(ent_val)
                ib_res['pct_shuffled'].append(pct)
                ib_res['sample_number'].append(n)
                if enc_type == 'nl':
                    ib_res['lang_id'].append(id)
                    ib_res['lang_name'].append(lang_name)
                else:
                    ib_res['ib'].append(ib_val)
                    ib_res['beta'].append(id)

    return shuffled_encs_dict, ib_res


if __name__ == '__main__':

    # Imports
    import pickle as pkl
    from empirical_tests.get_convexity_from_encoders import load_wcs_enc_data

    # Specify data load arguments
    enc_type = 'nl'  # synthetic or nl
    vz = 'optimal' # date_number if synthetic, str(number) if nl, 'optimal' if using pre-computed optimal ib encoders
    wcs_path = '/put wcs data directory path here'
    cielab_path = 'put cnum_CIELAB_table.txt path here'
    fig_path = 'put figures path here'

    # Load the relevant encoder data
    ib_res, cielab_map_df, enc_dict = load_wcs_enc_data(enc_type, vz, wcs_path, cielab_path)

    # Load the model data (needed to generate probabilitic need distribution for IB calculations)
    model_path = 'put IB_color_naming.pkl path here'
    with open(model_path, 'rb') as f:
        model_data = pkl.load(f)
    pm = model_data['pM'].flatten()

    # Shuffle the encoders, generate the IB results, and save
    p_vec = [round(0.05*(n+1), 2) for n in range(20)]
    shuffled_encs_dict, ib_res = generate_shuffled_encs(
        enc_dict, shuffle_pcts=p_vec, enc_type=enc_type, n_samples=3, pm=pm, verbose=True
    )

    # Convert IB results to dataframe and save as csv
    ib_res = pd.DataFrame(ib_res)
    res_path = f'/res/ib_results/nl_shuffled_enc_res.csv'
    ib_res.to_csv(res_path)

    # Save encoders in compatible format
    encoder_path = f'/res/encoders/nl_shuffled_enc.pkl'
    with open(encoder_path, 'wb') as f:
        pkl.dump(shuffled_encs_dict, f)