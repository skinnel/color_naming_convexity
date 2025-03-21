""" Script to create encoders from the WCS data in order to determine the complexity, accuracy and IB values for the
natural languages, as they occur in the World Color Survey (WCS). The output is a .csv with the aforementioned values
for each language, to be used in generating the IB chart.

@authors Lindsay Skinner (skinnel@uw.edu)
"""

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
from empirical_tests.wcs_data_processing import wcs_data_pull, create_color_space_table, lang_id_map, \
    create_lang_table, create_lang_encoder_dist
#from ib_from_scratch.ib_color_model_with_annealing import update_qw, update_mwu, drop_unused_dimensions, update_pu
#from ib_from_scratch.ib_with_zas_measures_test import update_qw, update_mwu, drop_unused_dimensions, update_pu
#from ib_from_scratch.ib_utils import get_entropy, get_mutual_information
from ib_from_scratch.zas_ib_utils import H, MI, joint, update_qw, update_mwu, update_pu

# Example using my IB test

# Pull the data
chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
    wcs_data_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/WCS-Data-20110316',
    CIELAB_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
)

# Generate meaning distributions, as defined in IB color paper
u_vals = cielab_df['chip_id'].values
meanings = generate_WCS_meanings(perceptual_variance=64)
pm = np.ones(meanings.shape[0]) / meanings.shape[0]
#pum_joint = meanings / meanings.sum()
#pu_m = pum_joint.T / pm
pu_m = meanings

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
    # mwu_df = np.ones((enc_df.shape[1], u_vals.shape[0]))
    #mwu_df = np.ones((u_vals.shape[0], enc_df.shape[1]))
    #qw_df, enc_df, mwu_df = drop_unused_dimensions(qw_df, enc_df, mwu_df, eps=0.001)
    mwu_df = update_mwu(qw_df, pm, pu_m, enc_df)
    #qw_df, enc_df, mwu_df = drop_unused_dimensions(qw_df, enc_df, mwu_df, eps=0.001)
    pu_df = mwu_df.sum(axis=1)
    #pu_df = qw_df @ mwu_df.T
    #pu_df = mwu_df @ qw_df
    pu_df = update_pu(mwu_df, qw_df, enc_df, meanings)

    # Get accuracy, complexity and ib values
    #acc_joint = joint(mwu_df, pu_df)
    acc_joint = joint(mwu_df.T, qw_df)
    #comp_joint = joint(enc_df, qw_df)
    comp_joint = joint(enc_df.T, pm)
    acc_val = MI(acc_joint)
    comp_val = MI(comp_joint)
    ent_val = H(qw_df)

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
res_path = f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/ib_results/wcs_zas_res3.csv'
wcs_ib_df.to_csv(res_path)

# Save encoders
encoder_path = f'/Users/lindsayskinner/PycharmProjects/clmsThesis/res/encoders/wcs_zas_encoders3.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(wcs_encoders, f)

# Create IB-like plot of natural languages
complexity = wcs_ib_df['complexity']
accuracy = wcs_ib_df['accuracy']
plt_path = f'/Users/lindsayskinner/Documents/school/CLMS/Thesis/figures/ibplot_wcs_zas3.png'
plt.scatter(complexity, accuracy, marker="o")
# save figure
plt.savefig(plt_path)

