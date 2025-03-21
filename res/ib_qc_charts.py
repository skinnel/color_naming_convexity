""" This script generates the quasi-convexity charts for the IB color model results."""

# Imports
import pandas as pd
import matplotlib
from os import listdir

# Specify results file
#res_file = 'res/auto_mesh/ib_qc_results_2_0.csv'
#res_file = 'all_auto'
#res_file = 'all_5'
#res_file = 'all_1'
#res_file = 'select_1'
res_file = 'anneal'

# Import results
if res_file == 'all_auto':
    res_path = 'res/auto_mesh/'
    all_files = listdir(res_path)
    res_df = None
    for f in all_files:
        if res_df is None:
            res_df = pd.read_csv(f'{res_path}/{f}')
        else:
            new_df = pd.read_csv(f'{res_path}/{f}')
            res_df = pd.concat([res_df, new_df])
elif res_file == 'all_5':
    res_path = 'res/fixed_mesh/five_perc/'
    all_files = listdir(res_path)
    res_df = None
    for f in all_files:
        if res_df is None:
            res_df = pd.read_csv(f'{res_path}/{f}')
        else:
            new_df = pd.read_csv(f'{res_path}/{f}')
            res_df = pd.concat([res_df, new_df])
elif res_file == 'all_1':
    res_path = 'res/fixed_mesh/one_perc/'
    all_files = listdir(res_path)
    res_df = None
    for f in all_files:
        if res_df is None:
            res_df = pd.read_csv(f'{res_path}/{f}')
        else:
            new_df = pd.read_csv(f'{res_path}/{f}')
            res_df = pd.concat([res_df, new_df])
elif res_file == 'select_1':
    res_path = 'res/fixed_mesh/one_perc/'
    all_files = listdir(res_path)
    res_df = None
    for f in all_files:
        if f.startswith('ib_qc_results_1_0'):
            if res_df is None:
                res_df = pd.read_csv(f'{res_path}/{f}')
            else:
                new_df = pd.read_csv(f'{res_path}/{f}')
                res_df = pd.concat([res_df, new_df])
elif res_file == 'anneal':
    res_path = 'res/annealing2/'
    all_files = listdir(res_path)
    res_df = None
    for f in all_files:
        if res_df is None:
            res_df = pd.read_csv(f'{res_path}/{f}')
        else:
            new_df = pd.read_csv(f'{res_path}/{f}')
            res_df = pd.concat([res_df, new_df])
else:
    res_df = pd.read_csv(res_file)

# Plot accuracy vs. quasi-convexity
acc_fig = res_df.plot.scatter(x='accuracy', y='convexity')

# Plot complexity vs. quasi-convexity
comp_fig = res_df.plot.scatter(x='complexity', y='convexity')

# Plot IB objective vs. quasi-convexity
obj_fig = res_df.plot.scatter(x='objective', y='convexity')

# Heatmap of complexity vs. accuracy with IB objective as temperature
main_fig_obj = res_df.plot.scatter(x='accuracy', y='complexity', c='objective', colormap='viridis')

# Heatmap of complexity vs. accuracy with quasi-convexity as temperature
main_fig_conv = res_df.plot.scatter(x='accuracy', y='complexity', c='convexity', colormap='viridis')