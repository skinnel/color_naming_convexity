""" Script to run the convexity evaluation for suboptimal, intermittent solutions obtained from my IB-color naming
model."""

# Imports
import numpy as np
import pandas as pd
from ib_from_scratch.my_ib_color_model import IBModel
from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from empirical_tests.utils import get_quasi_concavity_measure
from empirical_tests.wcs_data_processing import wcs_data_pull

# Set values for annealing.
#beta = 1.09
n_beta = 1000  # 1500
min_beta = 1.0
max_beta = 200  # 2.0**13 # beta too large - causes inf value in encoder calc # 50, 10, 5 also too large
n_words = 330
n_iter = 500
iter_step = 1
anneal_direction = 'descend'  # ascend or descend
beta_diff = 'subtraction'  # division or subtraction
#res_path = f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/ib_qc_results.csv'

# Calculate the beta values
assert n_beta > 0, 'n_beta must be a positive integer'
if n_beta == 1:
    beta_vals = [(min_beta + max_beta) / 2]
else:
    if beta_diff == 'subtraction':
        beta_vals = []
        mesh = (max_beta - min_beta) / (n_beta - 1)
        if anneal_direction == 'ascend':
            cur_beta = min_beta
            while cur_beta <= max_beta:
                beta_vals.append(cur_beta)
                cur_beta += mesh
        else:
            cur_beta = max_beta
            while cur_beta >= min_beta:
                beta_vals.append(cur_beta)
                cur_beta -= mesh
    else:
        beta_vals = []
        beta_i = 0
        cur_beta = max_beta
        beta_mesh = None
        while beta_i <= n_beta:
            beta_vals.append(cur_beta)
            if cur_beta > 2.0:
                cur_beta = (cur_beta + min_beta) / 2.0
            else:
                if beta_mesh == None:
                    beta_mesh = (cur_beta - min_beta) / (n_beta - beta_i)
                cur_beta -= beta_mesh
            beta_i += 1
        if anneal_direction == 'ascend':
            beta_vals = beta_vals.reverse()

# Pull the data.
chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
    wcs_data_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/WCS-Data-20110316',
    CIELAB_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
)

# Initialize color space values.
u_vals = cielab_df['chip_id'].values
u_coords = cielab_df[['L', 'A', 'B']].values
words = np.array(range(n_words))

# Generate meaning distributions, as defined in IB color paper.
meanings = generate_WCS_meanings(perceptual_variance=64)

# Initialize IB Model.
ibm = IBModel(observations=u_vals, coordinates=u_coords, meanings=meanings, words=words)
ibm.randomly_initialize(rseed=42, uniform=False)
#ibm.one_to_one_initialize(old_method=False)

# Perform beta annealing
for beta in beta_vals:

    print(f'Processing beta = {beta}')

    beta_name = round(beta, 4)
    beta_name = str(beta_name)
    beta_name = beta_name.replace('.', '_')
    res_path = f'/res_original/annealing_{anneal_direction}/ib_qc_results_{beta_name}.csv'

    # Fit model and save results for each iteration.
    qc_res = {'iteration': [], 'objective': [], 'accuracy': [], 'complexity': [], 'convexity': []}
    for i in range(n_iter):

        # Update encoder.
        assert iter_step > 0, 'Iteration step size must be a positive number.'
        if iter_step == 1:
            single_res = ibm.run_single_annealing_iteration(beta=beta, return_obj_fun_value=True, old_method=False, verbose=False)
        else:
            multi_res = ibm.perform_beta_specific_annealing(beta=beta, thresh=None, max_iter=iter_step, verbose=False)

        # Calculate Quasi-Convexity.
        aggregate_qc = 0
        x_coords = ibm.u_coords


        # Get the quasi-convexity measure of the distribution defined by each word
        qc_vec = np.apply_along_axis(lambda x: get_quasi_concavity_measure(x_probs=x, x_coordinates=x_coords, mesh=0.001),
                                     axis=1, arr=ibm.fitted_encoder)
        qc_values = {wi: qc_vec[wi] for wi in range(len(qc_vec))}

        # Determine how many chips assign the highest probability to each color term in this color naming system.
        max_probs = np.amax(ibm.fitted_encoder, axis=1)
        chip_ct_vec = np.sum(ibm.fitted_encoder >= max_probs[:, None], axis=1)


        # Calculate a weighted aggregate of the quasi-convexity measure.
        # chip_ct_vec = np.array(chip_ct_vec)
        # qc_vec = np.array(qc_vec)
        chip_ct_vec = chip_ct_vec / chip_ct_vec.sum()
        aggregate_qc = np.sum(chip_ct_vec * qc_vec)

        # Store results
        qc_res['iteration'].append(i)
        qc_res['objective'].append(ibm.obj_fun_val)
        qc_res['accuracy'].append(ibm.accuracy)
        qc_res['complexity'].append(ibm.complexity)
        qc_res['convexity'].append(aggregate_qc)

        print(f'Finished iteration {i}: objective = {ibm.obj_fun_val}, quasi-concavity = {aggregate_qc}')

    # Write results to an external file
    qc_res_df = pd.DataFrame(qc_res)
    qc_res_df.to_csv(res_path)

