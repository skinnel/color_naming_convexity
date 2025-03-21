""" Script to run the convexity evaluation for suboptimal, intermittent solutions obtained from my IB-color naming
model."""

# Imports
import numpy as np
import pandas as pd
from ib_from_scratch.my_ib_color_model import IBModel
from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from empirical_tests.utils import get_quasi_concavity_measure
from empirical_tests.wcs_data_processing import wcs_data_pull

# Set values for annealing round.
beta = 1.09
n_words = 100
n_iter = 200
iter_step = 1
#res_path = f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/ib_qc_results.csv'

beta_name = str(beta)
beta_name = beta_name.replace('.', '_')
res_path = f'/res_original/fixed_mesh/one_perc/ib_qc_results_{beta_name}.csv'

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
    # qc_values = {}
    # qc_vec = []
    # chip_ct_vec = []
    x_coords = ibm.u_coords

    # TODO: Remove below after testing
    # for wi in range(words.shape[0]):
    #
    #     # Get probabilities and coordinates.
    #     x_probs = ibm.fitted_encoder[:, wi]
    #
    #     # Get the quasi-convexity measure of the distribution defined by the specified term.
    #     word_qc = get_quasi_concavity_measure(x_coordinates=x_coords, x_probs=x_probs, mesh=None)
    #     qc_values[wi] = word_qc
    #
    #     # Determine how many chips assign the highest probability to this color term in this color naming system.
    #     max_probs = np.amax(ibm.fitted_encoder, axis=1)
    #     chip_ct = sum(x_probs >= max_probs)
    #     chip_ct_vec.append(chip_ct)
    #     qc_vec.append(word_qc)

    # Get the quasi-convexity measure of the distribution defined by each word
    qc_vec = np.apply_along_axis(lambda x: get_quasi_concavity_measure(x_probs=x, x_coordinates=x_coords, mesh=0.01),
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

# Questions:

# 1. Do we need to be concerned about the fact that this set up requires us to specify the number of words ahead of
#   time? I had hoped that we could deal with this by just allowing a large number of words and assuming that the
#   optimization process would cause the majority to have 0 probability, but I haven't seen that happen in practice.
#   Though it could be that I'm not running for enough iterations (attempts so far have been 1000 iterations or fewer).

    # 3 words and 100 iterations: accuracy = 1.1764, complexity = 1.4222
        # all color terms have significant number of chips
    # 3 words and 1000 iterations: accuracy = 1.3991, complexity = 1.4958
        # all color terms still have sign num chips

    # 5 words and 100 iterations: accuracy = 2.3582, complexity = 2.0909
        # all color terms have sig num chips
    # 5 words and 1000 iterations: accuracy = 2.2191, complexity = 2.0760
        # more spread (19-106) but still sig num chips for all terms

    # 10 words and 100 iterations: accuracy = 3.1981, complexity = 2.4629
        # all color terms have sig num chips, but variation is wider
    # 10 words and 1000 iterations: accuracy = 2.9400, complexity = 2.4961
        # Only 7 words have chips associated with them

# 2. Accuracy and complexity measures seem unusually close. Double check assumption about p(m) being linguistic need vs.
#   some other probability. If this isn't the issue, try to figure out what's causing the problem.

# 3. Ask about the qc weighting discussion that we had last time.