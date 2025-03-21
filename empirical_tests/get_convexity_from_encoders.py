""" Generates the convexity (AKA quasi-concavity) measures for a set of encoders and, optionally, affiliated charts
comparing the IB objective function, accuracy, and complexity against the quasi-convexity."""

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from empirical_tests.wcs_data_processing import wcs_data_pull, create_color_space_table
from empirical_tests.utils import get_quasi_concavity_measure
from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from typing import Tuple, Dict
from scipy.spatial.distance import cdist


# Define function to load the WCS and encoder data
def load_wcs_enc_data(enc_type: str, vz: str, wcs_path: str, cielab_path: str, encoder_path: str = None,
                      ib_res_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ Loads the relevant WCS and encoder data in order to calculate the associated quasi-convexity measures.
    
    Arguments:
    ----------
    enc_type
        Specifies which type of encoder to load, synthetic (ib generated) or nl (wcs generated)
    vz
        The version of encoders to be loaded.
    wcs_path
        The path for the WCS data file.
    cielab_path
        The path for the CIELAB data file.
    encoder_path
        The path for the encoder pickle files. If none, the path is generated using the enc_type and vz values.
    ib_res_path
        The path for the ibs values. If none, the path is generated using the enc_type and vz values.
    
    Return:
    -------
    ib_res
        The dataframe containing the ib results.
    cielab_map_df
        The dataframe containing the CIELAB data.
    enc_dict
        The dictionary containing the encoder dataframes. 
    """

    # Load IB-generated encoders
    if enc_type == 'synthetic':

        if vz == 'optimal':
            # Load the optimal encoders
            if encoder_path is None:
                encoder_path = f'/res_original/encoders/optimal_enc.pkl'
            with open(encoder_path, 'rb') as f:
                enc_dict = pickle.load(f)

            # Load the pre-calculated IB, accuracy and complexity values for optimal encoders
            if ib_res_path is None:
                ib_res_path = f'/res_original/ib_results/optimal_enc_res.csv'
            ib_res = pd.read_csv(ib_res_path)

        elif vz == 'shuffled':
            # Load the shuffled encoders
            if encoder_path is None:
                encoder_path = f'/res_original/encoders/shuffled_enc.pkl'
            with open(encoder_path, 'rb') as f:
                enc_dict = pickle.load(f)

            # Load the pre-calculated IB, accuracy and complexity values for shuffled encoders
            if ib_res_path is None:
                ib_res_path = f'/res_original/ib_results/shuffled_enc_res.csv'
            ib_res = pd.read_csv(ib_res_path)

        else:
            # Load the calculated encoders
            if encoder_path is None:
                encoder_path = f'/res_original/encoders/encoders_{vz}.pkl'
            with open(encoder_path, 'rb') as f:
                enc_dict = pickle.load(f)

            # Load the pre-calculated IB, accuracy and complexity values from ib annealing
            if ib_res_path is None:
                ib_res_path = f'/res_original/ib_results/res_{vz}.csv'
            ib_res = pd.read_csv(ib_res_path)

    # Load natural language encoders from the WCS
    else:

        if vz == 'shuffled':

            # Load the encoders
            if encoder_path is None:
                encoder_path = f'/res_original/encoders/nl_shuffled_enc.pkl'
            with open(encoder_path, 'rb') as f:
                enc_dict = pickle.load(f)

            # Load the pre-calculated IB, accuracy and complexity values
            if ib_res_path is None:
                ib_res_path = f'/res_original/ib_results/nl_shuffled_enc_res.csv'
            ib_res = pd.read_csv(ib_res_path)

        else:

            # Load the encoders
            if encoder_path is None:
                encoder_path = f'/res_original/encoders/wcs_encoders.pkl'
            with open(encoder_path, 'rb') as f:
                enc_dict = pickle.load(f)

            # Load the pre-calculated IB, accuracy and complexity values
            if ib_res_path is None:
                ib_res_path = f'/res_original/ib_results/wcs_res.csv'
            ib_res = pd.read_csv(ib_res_path)

    # Get the coordinates specific to each color chip
    chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(wcs_data_path=wcs_path, CIELAB_path=cielab_path)
    cielab_map_df = create_color_space_table(cielab_df=cielab_df, chip_df=chip_df)

    return ib_res, cielab_map_df, enc_dict


# Define function to calculate convexity measure for a given encoder
def get_encoder_qc(
        encoder: pd.DataFrame, cielab_map: pd.DataFrame, mesh: float = None
) -> Tuple[float, Dict[str, float]]:
    """ Returns the aggregate QC value for the encoder, as well as the color-specific QC values.

    Arguments:
    ---------
    encoder
        The dataframe containing the encoder information. Each row corresponds to a color term (word) and each column
        corresponds to a color chip. The values indicate the probability of the use of that color term corresponding to
        that color chip.
    cielab_map
        The dataframe containing the CIELAB chip ids and coordinates.
    mesh
        A float value indicating the size of jump between probability values that should be used to specify the level
        sets. If mesh is None then we use a dynamic mesh value so that we calculate a different level set for each
        observed probability value in the data provided. This case requires using a weighted average to calculate the
        final metric, where weights are proportional to the dynamic mesh values.

    Returns:
    --------
    qc_val
        The overall quasi-convexity measure for the encoder.
    qc_dict
        The dictionary containing the color-term specific quasi-convexity measures. Keyed by color term number.
    """

    # Get chip coordinates
    chip_coords = cielab_map[['L', 'A', 'B']].values

    # Get the color-term specific quasi-convexity measures
    qc_values = {}
    aggregate_qc = 0.0
    for wi in range(encoder.shape[1]):
        chip_probs = encoder[:, wi]
        color_qc = get_quasi_concavity_measure(x_coordinates=chip_coords, x_probs=chip_probs, mesh=mesh)
        qc_values[wi] = color_qc

        # Determine how many chips assign this color term as most likely
        max_probs = np.amax(encoder, axis=1)
        chip_vec = chip_probs >= max_probs

        # Us chip count and number of color terms to determine
        #encoder_max_ct = encoder.apply(lambda x: x >= max_probs, axis=1)
        max_probs_arr = np.tile(max_probs, (encoder.shape[1], 1))
        max_probs_arr = max_probs_arr.transpose()
        encoder_max_ct = np.greater_equal(encoder, max_probs_arr)
        chip_adjust = encoder_max_ct.sum(axis=1)
        chip_vec = chip_vec / chip_adjust
        chip_ct = sum(chip_vec)
        wt = chip_ct/chip_probs.shape[0]

        # Update aggregate quasi-convexity measure for the color naming system.
        aggregate_qc += wt * color_qc

    return aggregate_qc, qc_values


# Function to add quasi-concavity values to ib results
def generate_ibs_qc_df(
        enc_dict: dict, cielab_map: pd.DataFrame, ib_res: pd.DataFrame, mesh: float = None, ordered: bool = True,
        tol: float = 0.00001, shuffled_encs: bool = False, enc_type: str = 'synthetic'
) -> Tuple[pd.DataFrame, list]:
    """ Generates the ib-qc dataframe, as well as the color-specific quasi-concavity values for each color term within
    each encoder contained in enc_dict.

    Arguments:
    ----------
    enc_dict
        The dictionary containing the probabilities and ib information for each encoder.
    cielab_map
        The dataframe containing the CIELAB chip ids and coordinates.
    ib_res
        The dataframe corresponding to enc_dict that contains the ib, complexity and accuracy information for the
        encoders.
    mesh
        A float value indicating the size of jump between probability values that should be used to specify the level
        sets. If mesh is None then we use a dynamic mesh value so that we calculate a different level set for each
        observed probability value in the data provided. This case requires using a weighted average to calculate the
        final metric, where weights are proportional to the dynamic mesh values.
    ordered
        Indicates whether ib_res and enc_dict are identically ordered. If not, tol must be provided in order to ensure
        the encoders and ib results can be matched on their beta values.
    tol
        Tolerance value used to check for beta equivalence when matching up values from ib_res with the qc values
        calculated from enc_dict.
    shuffled_incs
        If true, indicates the encoders being processed are the shuffled encoders (which can have multiple encoders for
        a single beta value). If false we assume a unique beta value for each encoder.
    enc_type
        Indicates if the shuffled encoders are being generated from synthetic encoders ('synthetic'), which have beta and
        ib values, or from the natural languages ('nl'), which have language ids and names.

    Returns:
    --------
    ib_qc_df
        The ib results dictionary with quasi-concavity values appended in a separate column
    enc_color_qc
        Dictionary containing the color-specific quasi-concavity values for each color term in a given encoder.
    """

    # Ensure the user-provided data is correctly associated
    assert ordered or tol, 'If ordered is false then tol must be provided'
    assert len(enc_dict['encoder']) == ib_res.shape[0], 'enc_dict and ib_res must contain the same number of encoders'
    if enc_type == 'nl':
        assert len(enc_dict['lang_id']) == ib_res.shape[0], 'enc_dict must contain the same set of lang_ids as ib_res'
    else:
        assert len(enc_dict['beta']) == ib_res.shape[0], 'enc_dict must contain the same set of beta values as ib_res'

    # Get ids
    if shuffled_encs:
        id_vec = enc_dict['id']
    elif enc_type == 'nl':
        id_vec = enc_dict['lang_id']
    else:
        id_vec = enc_dict['beta']

    # Iterate over the encoder dictionary
    print('Initializing new DF column.')
    ib_qc_df = ib_res.copy(deep=True)
    ib_qc_df['qc'] = 0.0
    #qc_dict = {}
    enc_color_qc = []
    for i in range(len(id_vec)):

        # Get the identifying value
        id_val = id_vec[i]

        # Get the specific encoder data
        print(f'Getting encoder data for step {i}/{len(id_vec)}')
        if enc_type == 'nl':
            lang_id = enc_dict['lang_id'][i]
            lang_name = enc_dict['lang_name'][i]
        else:
            b = enc_dict['beta'][i]
        encoder = enc_dict['encoder'][i]
        encoder = encoder.transpose()

        # Calculate the quasi-concavity value
        print(f'Getting QC for step {i}/{len(id_vec)}')
        qc, color_dict = get_encoder_qc(encoder, cielab_map, mesh)

        # Update the results dataframe
        print(f'Updating results DF for step {i}/{len(id_vec)}')
        if ordered:
            ib_qc_df['qc'].iloc[i] = qc
        else:
            if shuffled_encs or enc_type == 'nl':
                ib_res_ids = list(ib_qc_df['id'])
                idx = ib_res_ids.index(id_val)
            else:
                idx = ib_qc_df.index[ib_qc_df['beta'] - b >= tol]
            ib_qc_df.loc[idx, 'qc'] = qc

        # Save color-specific quasi-concavity results
        enc_color_qc.append(color_dict)

    return ib_qc_df, enc_color_qc


# Function to generate the ib-qc charts
def create_charts(res_df: pd.DataFrame, dir_path: str, enc_type:str, vz: str, id: str):
    """Creates the ib/accuracy/complexity vs qc charts and ib frontier with qc as temperature for the provided 
    dataframe. Saves the charts in the specified directory.
    
    Arguments:
    ----------
    res_df
        A dataframe containing the accuracy, complexity, IB value and quasi-concavity values in separate columns.
    dir_path
        The file path to the director that will contain the charts.
    enc_type
        The encoder type to specify whether ib charts can be generated.
    vz
        The version identifier for the charts.
    id
        The unique run identifier for the charts.
    """

    if enc_type == 'nl':
        vz = f'{vz}_NL'

    # Generate accuracy vs qc plot
    acc_plt_path = f'{dir_path}/acc_{vz}_{id}.png'
    acc_fig = res_df.plot.scatter(x='accuracy', y='qc').get_figure()
    acc_fig.savefig(acc_plt_path)

    # Generate complexity vs qc plot
    comp_plt_path = f'{dir_path}/comp_{vz}_{id}.png'
    comp_fig = res_df.plot.scatter(x='complexity', y='qc').get_figure()
    comp_fig.savefig(comp_plt_path)

    # Generate qc vs ib plot
    if enc_type != 'nl':
        ib_plt_path = f'{dir_path}/ib_{vz}_{id}.png'
        ib_fig = res_df.plot.scatter(x='ib', y='qc').get_figure()
        ib_fig.savefig(ib_plt_path)

    # Generate ib frontier with qc indicated by color
    ib_qc_plt_path = f'{dir_path}/ib_qc_{vz}_{id}.png'
    ib_qc_fig = res_df.plot.scatter(x='complexity', y='accuracy', c='qc', colormap='viridis').get_figure()
    ib_qc_fig.savefig(ib_qc_plt_path)


# Function to generate the ib-qc charts
def create_aggregate_charts(res_df: pd.DataFrame, dir_path: str, vz: str, id: str, enc_type: str):
    """Creates the ib/accuracy/complexity vs qc charts and ib frontier with qc as temperature for the provided
    dataframe. Saves the charts in the specified directory.

    Arguments:
    ----------
    res_df
        A dataframe containing the aggregate ib results in separate columns.
    dir_path
        The file path to the director that will contain the charts.
    vz
        The version identifier for the charts.
    id
        The unique run identifier for the charts.
    enc_type
        The encoder type to specify whether beta or lang_id is the core identifying column.
    """

    # Add new columns marking QC and QC diff as a percentage of the optimal QC
    res_df['qc_pct'] = None
    res_df['qc_diff_pct'] = None
    if enc_type == 'nl':
        id_vec = list(res_df['lang_id'].unique())
    else:
        id_vec = list(res_df['beta'].unique())

    for id_val in id_vec:
        if enc_type == 'nl':
            tmp_df = res_df[res_df['lang_id'] == id_val]
        else:
            tmp_df = res_df[res_df['beta'] == id_val]
        optimal_qc = tmp_df[tmp_df['pct_shuffled'] == 0.0]['qc'].values[0]
        if optimal_qc > 0.0:
            idx = tmp_df.index
            res_df.loc[idx, 'qc_pct'] = res_df.loc[idx, 'qc'] / optimal_qc
            res_df.loc[idx, 'qc_diff_pct'] = res_df.loc[idx, 'qc_diff'] / optimal_qc

    # Ensure unique filenames for the natural language charts
    if enc_type == 'nl':
        vz = f'{vz}_nl'

    # Generate qc vs frontier distance
    qc_dist_plt_path = f'{dir_path}/qc_dist_{vz}_{id}.png'
    qc_dist_fig = res_df.plot.scatter(x='frontier_dist', y='qc').get_figure()
    qc_dist_fig.savefig(qc_dist_plt_path)

    # Generate qc difference vs frontier distance
    qc_diff_dist_plt_path = f'{dir_path}/qc_diff_dist_{vz}_{id}.png'
    qc_diff_dist_fig = res_df.plot.scatter(x='frontier_dist', y='qc_diff').get_figure()
    qc_diff_dist_fig.savefig(qc_diff_dist_plt_path)

    # Generate qc percent vs frontier distance
    qc_pct_dist_plt_path = f'{dir_path}/qc_pct_dist_{vz}_{id}.png'
    qc_pct_dist_fig = res_df.plot.scatter(x='frontier_dist', y='qc_pct').get_figure()
    qc_pct_dist_fig.savefig(qc_pct_dist_plt_path)

    # Generate qc difference percent vs frontier distance
    qc_diff_pct_dist_plt_path = f'{dir_path}/qc_diff_pct_dist_{vz}_{id}.png'
    qc_diff_pct_dist_fig = res_df.plot.scatter(x='frontier_dist', y='qc_diff_pct').get_figure()
    qc_diff_pct_dist_fig.savefig(qc_diff_pct_dist_plt_path)

    # Generate qc vs percent shuffled
    qc_pct_plt_path = f'{dir_path}/qc_pct_{vz}_{id}.png'
    qc_pct_fig = res_df.plot.scatter(x='pct_shuffled', y='qc').get_figure()
    qc_pct_fig.savefig(qc_pct_plt_path)

    # Generate qc difference vs percent shuffled
    qc_diff_pct_plt_path = f'{dir_path}/qc_diff_pct_{vz}_{id}.png'
    qc_diff_pct_fig = res_df.plot.scatter(x='pct_shuffled', y='qc_diff').get_figure()
    qc_diff_pct_fig.savefig(qc_diff_pct_plt_path)

    # Generate qc vs percent shuffled
    qc_pct_pct_plt_path = f'{dir_path}/qc_pct_pct_{vz}_{id}.png'
    qc_pct_pct_fig = res_df.plot.scatter(x='pct_shuffled', y='qc_pct').get_figure()
    qc_pct_pct_fig.savefig(qc_pct_pct_plt_path)

    # Generate qc difference vs percent shuffled
    qc_diff_pct_pct_plt_path = f'{dir_path}/qc_diff_pct_pct_{vz}_{id}.png'
    qc_diff_pct_pct_fig = res_df.plot.scatter(x='pct_shuffled', y='qc_diff_pct').get_figure()
    qc_diff_pct_pct_fig.savefig(qc_diff_pct_pct_plt_path)


def aggregate_shuffled_results(optimal_enc_res: pd.DataFrame, shuffled_enc_res: pd.DataFrame, enc_type: str) -> pd.DataFrame:
    """ Aggregates the shuffled encoder results (which contain multiple runs for each percentage shuffled for each beta
    value) and returns the average ib, qc, accuracy, complexity and approximate distance from the ib frontier for each
    (beta, percent-shuffled) pair.

    Arguments:
    ----------
    optimal_enc_res
        The ib-qc results for the optimal encoders, which define the ib-frontier in complexity X accuracy space.
    shuffled_enc_res
        The ib-qc results for the shuffled encoders.
    enc_type
        Indicates if the shuffled encoders are being generated from synthetic encoders ('synthetic'), which have beta and
        ib values, or from the natural languages ('nl'), which have language ids and names.

    Returns:
    --------
    aggregate_res
        The ib-qc results averages for the shuffled encoders, aggregated across the n samples taken when creating the
        shuffled encoders.
    """

    # Initialize the results dictionary
    if enc_type == 'nl':
        aggregate_res = {
            'lang_id': [],
            'pct_shuffled': [],
            'accuracy': [],
            'complexity': [],
            'qc': [],
            'qc_diff': [],
            'frontier_dist': []
        }

        individual_res = {
            'lang_id': [],
            'pct_shuffled': [],
            'accuracy': [],
            'complexity': [],
            'qc': [],
            'qc_diff': [],
            'frontier_dist': []
        }
    else:
        aggregate_res = {
            'beta': [],
            'pct_shuffled': [],
            'accuracy': [],
            'complexity': [],
            'ib': [],
            'qc': [],
            'ib_diff': [],
            'qc_diff': [],
            'frontier_dist': []
        }

        individual_res = {
            'beta': [],
            'pct_shuffled': [],
            'accuracy': [],
            'complexity': [],
            'ib': [],
            'qc': [],
            'ib_diff': [],
            'qc_diff': [],
            'frontier_dist': []
        }

    # Define how aggregate groups will be defined
    if enc_type == 'nl':
        lang_id_list = list(shuffled_enc_res['lang_id'].unique())
        n = len(lang_id_list)
    else:
        n = optimal_enc_res.shape[0]

    for i in range(n):

        if enc_type == 'nl':
            print(f'Processing results for {i}/{n} shuffled languages.')
        else:
            print(f'Processing results for {i}/{n} beta values.')

        # Get optimal values
        if enc_type == 'nl':
            lang_id = lang_id_list[i]
        else:
            b = optimal_enc_res['beta'][i]
            opt_ib = optimal_enc_res['ib'][i]

        # Only add optimal encoders if processing synthetic (not NL) data
        frontier_pts = optimal_enc_res[['complexity', 'accuracy']].to_numpy(copy=True)
        if enc_type != 'nl':
            opt_complexity = optimal_enc_res['complexity'][i]
            opt_accuracy = optimal_enc_res['accuracy'][i]
            opt_qc = optimal_enc_res['qc'][i]

            # Save the optimal values in the aggregate results
            aggregate_res['beta'].append(b)
            aggregate_res['ib'].append(opt_ib)
            aggregate_res['ib_diff'].append(0.0)
            aggregate_res['pct_shuffled'].append(0.0)
            aggregate_res['accuracy'].append(opt_accuracy)
            aggregate_res['complexity'].append(opt_complexity)
            aggregate_res['qc'].append(opt_qc)
            aggregate_res['qc_diff'].append(0.0)
            aggregate_res['frontier_dist'].append(0.0)

            # Save the optimal values in the independent results
            individual_res['beta'].append(b)
            individual_res['ib'].append(opt_ib)
            individual_res['ib_diff'].append(0.0)
            individual_res['pct_shuffled'].append(0.0)
            individual_res['accuracy'].append(opt_accuracy)
            individual_res['complexity'].append(opt_complexity)
            individual_res['qc'].append(opt_qc)
            individual_res['qc_diff'].append(0.0)
            individual_res['frontier_dist'].append(0.0)

        # Get the shuffled encoder results for the specified identifier value
        if enc_type == 'nl':
            shuffled_res_id = shuffled_enc_res.loc[shuffled_enc_res['lang_id'] == lang_id, :]
        else:
            shuffled_res_id = shuffled_enc_res.loc[shuffled_enc_res['beta'] == b, :]
        pct_vec = shuffled_enc_res['pct_shuffled'].unique()
        for pct in pct_vec:

            # Get the shuffled encoder results for the specified percent shuffled value
            shuffled_res_pct = shuffled_res_id.loc[shuffled_res_id['pct_shuffled'] == pct]
            shuffled_res_pct = shuffled_res_pct.copy(deep=True)
            #complexity_diff = shuffled_res_pct['complexity'] - opt_complexity
            #accuracy_diff = shuffled_res_pct['accuracy'] - opt_accuracy
            if enc_type != 'nl':
                ib_diff = shuffled_res_pct['ib'] - opt_ib
                qc_diff = shuffled_res_pct['qc'] - opt_qc
            else:
                ib_diff = shuffled_res_pct['qc'] * 0.0
                qc_diff = shuffled_res_pct['qc']
            #frontier_dist = (complexity_diff**2) + (accuracy_diff**2)
            #frontier_dist = frontier_dist**(0.5)
            shuffled_pts = shuffled_res_pct[['complexity', 'accuracy']].to_numpy(copy=True)
            dist_array = cdist(frontier_pts, shuffled_pts)
            frontier_dist = dist_array.min(axis=0)


            # Aggregate the results
            complexity_agg = shuffled_res_pct['complexity'].mean()
            accuracy_agg = shuffled_res_pct['accuracy'].mean()
            qc_agg = shuffled_res_pct['qc'].mean()
            qc_diff_agg = qc_diff.mean()
            frontier_dist_agg = frontier_dist.mean()
            if enc_type != 'nl':
                ib_agg = shuffled_res_pct['ib'].mean()
                ib_diff_agg = ib_diff.mean()

            # Save the aggregate results
            if enc_type == 'nl':
                aggregate_res['lang_id'].append(lang_id)
            else:
                aggregate_res['beta'].append(b)
                aggregate_res['ib'].append(ib_agg)
                aggregate_res['ib_diff'].append(ib_diff_agg)
            aggregate_res['pct_shuffled'].append(pct)
            aggregate_res['accuracy'].append(accuracy_agg)
            aggregate_res['complexity'].append(complexity_agg)
            aggregate_res['qc'].append(qc_agg)
            aggregate_res['qc_diff'].append(qc_diff_agg)
            aggregate_res['frontier_dist'].append(frontier_dist_agg)

            # Save the individual results
            if enc_type == 'nl':
                lang_id_vals = [lang_id] * shuffled_res_pct.shape[0]
                individual_res['lang_id'].extend(lang_id_vals)
            else:
                beta_vals = [b] * shuffled_res_pct.shape[0]
                individual_res['beta'].extend(beta_vals)
                ib_vals = list(shuffled_res_pct['ib'])
                individual_res['ib'].extend(ib_vals)
                individual_res['ib_diff'].extend(ib_diff)
            pct_vals = [pct] * shuffled_res_pct.shape[0]
            individual_res['pct_shuffled'].extend(pct_vals)
            acc_vals = list(shuffled_res_pct['accuracy'])
            individual_res['accuracy'].extend(acc_vals)
            comp_vals = list(shuffled_res_pct['complexity'])
            individual_res['complexity'].extend(comp_vals)
            qc_vals = list(shuffled_res_pct['qc'])
            individual_res['qc'].extend(qc_vals)
            individual_res['qc_diff'].extend(qc_diff)
            individual_res['frontier_dist'].extend(frontier_dist)

    print('Finished processing results.')

    # Convert results to a dataframe
    aggregate_res = pd.DataFrame(aggregate_res)
    individual_res = pd.DataFrame(individual_res)

    return aggregate_res, individual_res


def check_correlations(res_df: pd.DataFrame, enc_type: str) -> pd.DataFrame:
    """ Determines the correlations between the distance from the frontier and the QC and (if available) the QC diff
    values for each beta.

    Arguments:
    ----------
    res_df
        A dataframe containing the ib results and frontier distance for all shuffled results.
    enc_type
        Indicates if the shuffled encoders are being generated from synthetic encoders ('synthetic'), which have beta and
        ib values, or from the natural languages ('nl'), which have language ids and names.


    Returns:
    --------
    cor_df
        A dataframe showing the correlation values between the distance from the frontier and the QC and QC diff values,
        for each beta.
    """

    # Initialize results
    if enc_type == 'nl':
        cor_res = {
            'lang_id': [],
            'qc_dist_cor': [],
            'qc_acc_cor': [],
            'qc_comp_cor': []
        }
    else:
        cor_res = {
            'beta': [],
            'qc_dist_cor': [],
            'qc_acc_cor': [],
            'qc_comp_cor': []
        }

    # Get correlation values
    if enc_type == 'nl':
        id_vec = list(res_df['lang_id'].unique())
    else:
        id_vec = list(res_df['beta'].unique())
    for id in id_vec:
        if enc_type == 'nl':
            tmp_df = res_df[res_df['lang_id'] == id]
        else:
            tmp_df = res_df[res_df['beta'] == id]
        tmp_df = tmp_df.loc[tmp_df['pct_shuffled'] > 0.0, :]
        tmp1 = tmp_df[['frontier_dist', 'qc']]
        tmp2 = tmp_df[['frontier_dist', 'accuracy']]
        tmp3 = tmp_df[['frontier_dist', 'complexity']]
        cor_qc = tmp1.corr()
        cor_acc = tmp2.corr()
        cor_comp = tmp3.corr()

        # Save results
        if enc_type == 'nl':
            cor_res['lang_id'].append(id)
        else:
            cor_res['beta'].append(id)
        cor_res['qc_dist_cor'].append(cor_qc.iloc[0, 1])
        cor_res['qc_acc_cor'].append(cor_acc.iloc[0, 1])
        cor_res['qc_comp_cor'].append(cor_comp.iloc[0, 1])

    # Convert results to a dataframe
    cor_res = pd.DataFrame(cor_res)

    return cor_res


if __name__ == '__main__':

    # Specify data load arguments
    # synthetic or nl
    enc_type = 'nl'
    # date_number if synthetic, number if nl, 'optimal' if using pre-computed optimal ib encoders, 'shuffled' if using
    # the suboptimal shuffled encoders
    #vz = '20241207_8'
    vz = 'shuffled'
    id = '1'
    wcs_path = '/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/WCS-Data-20110316'
    cielab_path = '/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
    fig_path = '/Users/lindsayskinner/Documents/school/CLMS/Thesis/figures/qc_figures'

    # Load data
    ib_res, cielab_map_df, enc_dict = load_wcs_enc_data(enc_type, vz, wcs_path, cielab_path)

    # Generate QC-IB value pairings
    ib_qc_df, enc_color_qc = generate_ibs_qc_df(
        enc_dict, cielab_map_df, ib_res, mesh=0.01, ordered=True, enc_type=enc_type
    )

    # Save the ib-qc results
    if enc_type == 'nl':
        if vz == 'shuffled':
            ib_qc_path = f'/res_original/ib_qc_results/nl_{vz}_res.csv'
        else:
            ib_qc_path = f'/res_original/ib_qc_results/wcs_res.csv'
    else:
        ib_qc_path = f'/res_original/ib_qc_results/{vz}_res.csv'
    ib_qc_df.to_csv(ib_qc_path)

    # Generate and save colored plots of the results
    create_charts(res_df=ib_qc_df, dir_path=fig_path, vz=vz, id=id, enc_type=enc_type)

    # Aggregate the multiple samples if using shuffled encoders
    if vz == 'shuffled' or enc_type == 'nl':

        # Load results for IB Frontier encoders
        optimal_res = pd.read_csv('/res_original/ib_qc_results/optimal_res.csv')

        # Aggregate the results
        ib_qc_agg_df, ib_qc_all_shuffled_df = aggregate_shuffled_results(optimal_res, ib_qc_df, enc_type=enc_type)

        # # Save the aggregate ib-qc results
        # ib_qc_agg_path = f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/ib_qc_results/{vz}_res{id}_aggregate.csv'
        # ib_qc_agg_df.to_csv(ib_qc_agg_path)

        # Save the individual ib-qc results
        if enc_type == 'nl' and vz == 'shuffled':
            ib_qc_all_shuffled_path = (
                f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/ib_qc_results/nl_{vz}_res_all.csv'
            )
        elif enc_type == 'nl':
            ib_qc_all_shuffled_path = (
                f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/ib_qc_results/nl_res_all.csv'
            )
        else:
            ib_qc_all_shuffled_path = (
                f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/ib_qc_results/shuffled_res_all.csv'
            )
        ib_qc_all_shuffled_df.to_csv(ib_qc_all_shuffled_path)

        # # Generate plots for the aggregate results
        # agg_id = f'{id}_aggregate'
        # create_charts(res_df=ib_qc_agg_df, dir_path=fig_path, vz=vz, id=agg_id, enc_type=enc_type)
        #
        # # Additional aggregate plots
        # create_aggregate_charts(res_df=ib_qc_agg_df, dir_path=fig_path, vz=vz, id=agg_id, enc_type=enc_type)
        #
        # # Additional plots for individual shuffled samples
        # ind_id = f'{id}_all'
        # create_aggregate_charts(res_df=ib_qc_all_shuffled_df, dir_path=fig_path, vz=vz, id=ind_id, enc_type=enc_type)
        #
        # # Get correlations
        # cor_df = check_correlations(ib_qc_all_shuffled_df, enc_type=enc_type)
        # agg_cor_df = check_correlations(ib_qc_agg_df, enc_type=enc_type)
        #
        # # Save correlation results
        # all_cor_path = f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/qc_dist_correlations/qc_dist_cor_{vz}_{ind_id}.csv'
        # cor_df.to_csv(all_cor_path)
        # agg_cor_path = f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/qc_dist_correlations/qc_dist_cor_{vz}_{agg_id}.csv'
        # agg_cor_df.to_csv(agg_cor_path)
        #
        # # Check all correlations
        # if enc_type == 'nl':
        #     tmp_df = ib_qc_all_shuffled_df[['qc', 'frontier_dist', 'accuracy', 'complexity']].copy(deep=True)
        # else:
        #     beta_max = None  # Positive float value or None
        #     if beta_max:
        #         tmp_df = ib_qc_all_shuffled_df.loc[ib_qc_all_shuffled_df['beta'] < beta_max]
        #         tmp_df = tmp_df.loc[tmp_df['beta'] > 0.0]
        #     else:
        #         tmp_df = ib_qc_all_shuffled_df.loc[ib_qc_all_shuffled_df['beta'] > 0.0]
        #     tmp_df = tmp_df[['qc', 'frontier_dist', 'accuracy', 'complexity', 'ib']].copy(deep=True)
        # tmp_df.dropna(inplace=True)
        # # Get correlation against QC
        # tmp1 = tmp_df[['frontier_dist', 'qc']]
        # tmp2 = tmp_df[['accuracy', 'qc']]
        # tmp3 = tmp_df[['complexity', 'qc']]
        # if enc_type != 'nl':
        #     tmp4 = tmp_df[['ib', 'qc']]
        # cor_qc = tmp1.corr()
        # cor_acc = tmp2.corr()
        # cor_comp = tmp3.corr()
        # if enc_type != 'nl':
        #     cor_ib = tmp4.corr()
        # print(f'Cor(QC, Frontier Distance) = {cor_qc.iloc[0,1]}')
        # print(f'Cor(QC, Accuracy) = {cor_acc.iloc[0, 1]}')
        # print(f'Cor(QC, Complexity) = {cor_comp.iloc[0, 1]}')
        # if enc_type != 'nl':
        #     print(f'Cor(QC, IB) = {cor_ib.iloc[0, 1]}')
        # # Get all correlations
        # cor_all = tmp_df.corr()
        # if enc_type == 'nl':
        #     print('All correlations: qc, frontier distance, accuracy, complexity')
        # else:
        #     print('All correlations: qc, frontier distance, accuracy, complexity, ib')
        # print(cor_all)
        #
        # # Check regression model (relative correlations)
        # from sklearn.linear_model import LinearRegression
        # # First test - no IB
        # xvals = tmp_df[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
        # yvals = tmp_df['frontier_dist'].to_numpy(copy=True)
        # reg = LinearRegression(fit_intercept=False)
        # reg.fit(xvals, yvals)
        # print(f'Regression coefficients: qc, acc, comp = {reg.coef_}')
        # #print(f'Regression intercept: {reg.intercept_}')
        # if enc_type != 'nl':
        #     # Second test - with IB
        #     xvals = tmp_df[['qc', 'accuracy', 'complexity', 'ib']].to_numpy(copy=True)
        #     yvals = tmp_df['frontier_dist'].to_numpy(copy=True)
        #     reg = LinearRegression(fit_intercept=False)
        #     reg.fit(xvals, yvals)
        #     print(f'Regression coefficients: qc, acc, comp, ib = {reg.coef_}')
        #     #print(f'Regression intercept: {reg.intercept_}')
        #     # Third test - only frontier distance and IB
        #     xvals = tmp_df[['qc', 'ib']].to_numpy(copy=True)
        #     yvals = tmp_df['frontier_dist'].to_numpy(copy=True)
        #     reg = LinearRegression(fit_intercept=False)
        #     reg.fit(xvals, yvals)
        #     print(f'Regression coefficients: qc, ib = {reg.coef_}')
        #     #print(f'Regression intercept: {reg.intercept_}')



    # Future plans
    # 1. Be able to load multiple ib_qc results and combine into one
    # 2. Modify the chart creation so we can specify a line along the frontier, and scatterplots for other values
    # 3. [DONE] write a script to generate shuffled encoders from the optimal and nl encoders
