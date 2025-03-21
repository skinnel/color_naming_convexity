""" Script to create encoders from the WCS data in order to determine the complexity, accuracy and IB values for the
natural languages, as they occur in the World Color Survey (WCS). The output is a .csv with the aforementioned values
for each language, to be used in generating the IB chart.

@authors Lindsay Skinner (skinnel@uw.edu)
"""

import numpy as np
import pandas as pd
import pickle
from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from empirical_tests.wcs_data_processing import wcs_data_pull, create_color_space_table, lang_id_map, \
    create_lang_table, create_lang_encoder_dist
from ib_from_scratch.ib_color_model_with_annealing import update_qw, update_mwu, drop_unused_dimensions#, update_pu
from ib_from_scratch.ib_utils import get_entropy, get_mutual_information

# Example using my IB test

# Pull the data
chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
    wcs_data_path='put wcs directory path here',
    CIELAB_path='put cnum_CIELAB_table.txt path here'
)

# load the model data
model_path = 'put IB_color_naming.pkl path here'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# Separate the encoders, betas and pu
optimal_encs = model_data['qW_M']
optimal_betas = model_data['betas']
#pu_df = model_data['pM'].flatten()
pm = model_data['pM'].flatten()

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

# Get the complexity, accuracy and IB value for each natural language in WCS
lang_list = lang_df['lang_name'].to_list()
wcs_ib_results = {
    'lang_id': [],
    'lang_name': [],
    'complexity': [],
    'accuracy': [],
    'entropy': []
}
wcs_encoders = {'lang_id': [],
                'lang_name': [],
                'encoder': []}
for lang in lang_list:

    print(f'Processing: {lang}')

    # Set language id
    lang_id = lang_id_map(language=lang, lang_df=lang_df)

    # Process the data
    cielab_map_df = create_color_space_table(cielab_df=cielab_df, chip_df=chip_df)
    term_ct_df = create_lang_table(term_df=term_df, vocab_df=vocab_df, lang_id=lang_id)
    word_probs = create_lang_encoder_dist(term_ct_df=term_ct_df)
    enc_df = word_probs.iloc[:, 1:]
    enc_df = enc_df.transpose()
    enc_df = enc_df.to_numpy()

    # Get other required distributions
    qw_df = update_qw(pm, enc_df)
    mwu_df = update_mwu(qw_df, pm, pu_m, enc_df)
    qw_df, enc_df, mwu_df = drop_unused_dimensions(qw_df, enc_df, mwu_df, eps=0.001)

    # Get accuracy, complexity and ib values
    acc_val = get_mutual_information(pu, qw_df, mwu_df)
    comp_val = get_mutual_information(qw_df, pm, enc_df)
    ent_val = get_entropy(qw_df)

    # Save IB results
    wcs_ib_results['lang_id'].append(lang_id)
    wcs_ib_results['lang_name'].append(lang)
    wcs_ib_results['complexity'].append(comp_val)
    wcs_ib_results['accuracy'].append(acc_val)
    wcs_ib_results['entropy'].append(ent_val)

    # Save encoders in correct format for convexity assessment
    wcs_encoders['lang_id'].append(lang_id)
    wcs_encoders['lang_name'].append(lang)
    wcs_encoders['encoder'].append(enc_df)

# Convert IB results to dataframe and save as csv
wcs_ib_df = pd.DataFrame(wcs_ib_results)
res_path = f'/res/ib_results/wcs_res.csv'
wcs_ib_df.to_csv(res_path)

# Save encoders
encoder_path = f'/res/encoders/wcs_encoders.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(wcs_encoders, f)


