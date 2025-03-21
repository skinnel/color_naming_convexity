""" Pulls the optimal encoders defined in (Zaslavsky, 2018) in order to evaluate the quasi-concavity measure of those
encoders."""

# Libraries
import pickle as pkl
import numpy as np
import pandas as pd
from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from ib_from_scratch.ib_color_model_with_annealing import update_qw, update_mwu, drop_unused_dimensions
from ib_from_scratch.ib_utils import get_entropy, get_mutual_information

# load the model data
model_path = '/Users/lindsayskinner/Documents/school/CLMS/Thesis/IB_color_naming_model/IB_color_naming.pkl'
with open(model_path, 'rb') as f:
    model_data = pkl.load(f)

# Separate the encoders, betas and pu
optimal_encs = model_data['qW_M']
optimal_betas = model_data['betas']
#pu_df = model_data['pM'].flatten()
pm = model_data['pM'].flatten()

# Initialize results objects
ib_res = {
    'complexity': [],
    'accuracy': [],
    'entropy': [],
    'ib': [],
    'beta': []
}
enc_formatted = {
    'beta': [],
    'ib_val': [],
    'encoder': []
}

# Process the encoders to get the necessary outputs and format for QC calcuations and plot generation
meanings = generate_WCS_meanings(perceptual_variance=64)
#pm = np.ones(330) / 330
pu_m = meanings

# Get p(u) from p(m) and m(u) = p(u|m)
# TODO: check columns vs rows in for loop
pu = np.zeros(330)
for mi in range(pu_m.shape[0]):
    mu = pu_m[mi, :]
    wt_m = pm[mi]
    pu_delta = wt_m * mu
    pu = pu + pu_delta

for i in range(len(optimal_encs)):

    # Get encoder and beta
    enc = optimal_encs[i].transpose()
    b = optimal_betas[i]

    # Get other required distributions
    qw_df = update_qw(pm, enc)
    mwu_df = update_mwu(qw_df, pm, pu_m, enc)
    qw_df, enc_df, mwu_df = drop_unused_dimensions(qw_df, enc, mwu_df, eps=0.001)
    #pu_df = mwu_df.sum(axis=1)

    # Calculate results
    acc_val = get_mutual_information(pu, qw_df, mwu_df)
    comp_val = get_mutual_information(qw_df, pm, enc_df)
    ent_val = get_entropy(qw_df)
    ib_val = comp_val - (b * acc_val)

    # Save IB results
    ib_res['complexity'].append(comp_val)
    ib_res['accuracy'].append(acc_val)
    ib_res['entropy'].append(ent_val)
    ib_res['ib'].append(ib_val)
    ib_res['beta'].append(b)

    # Save encoder results
    enc_formatted['beta'].append(b)
    enc_formatted['ib_val'].append(ib_val)
    enc_formatted['encoder'].append(enc)

# Convert IB results to dataframe and save as csv
ib_res = pd.DataFrame(ib_res)
res_path = f'/res/ib_results/optimal_enc_res.csv'
ib_res.to_csv(res_path)

# Save encoders in compatible format
encoder_path = f'/res/encoders/optimal_enc.pkl'
with open(encoder_path, 'wb') as f:
    pkl.dump(enc_formatted, f)
