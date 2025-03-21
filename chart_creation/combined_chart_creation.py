""" Creates the charts that comnbine the various encoder types."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colormaps
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define path to final chart directory
chart_dir = 'put_chart_directory_here'

# Define encoder results paths
optimal_enc_res_path = 'res/ib_qc_results/optimal_res.csv'
shuffled_enc_res_path = 'res/ib_qc_results/shuffled_res_all.csv'
nl_enc_res_path = 'res/ib_qc_results/wcs_res.csv'
nl_shuffled_enc_res_path = 'res/ib_qc_results/nl_shuffled_res_all.csv'

# Get results dataframes
optimal_enc = pd.read_csv(optimal_enc_res_path)
shuffled_enc = pd.read_csv(shuffled_enc_res_path)
nl_enc = pd.read_csv(nl_enc_res_path)
nl_shuffled_enc = pd.read_csv(nl_shuffled_enc_res_path)

# Remove optimal encoders from the shuffled encoders dataframe
shuffled_enc = shuffled_enc[shuffled_enc['frontier_dist'] > 0.0]

# Add column to NL encoders that contains distance from the ib frontier
frontier_pts = optimal_enc[['complexity', 'accuracy']].to_numpy(copy=True)
nl_pts = nl_enc[['complexity', 'accuracy']].to_numpy(copy=True)
dist_array = cdist(frontier_pts, nl_pts)
frontier_dist = dist_array.min(axis=0)
nl_enc['frontier_dist'] = frontier_dist

# Add column to optimal encoders that contains distance from the ib frontier
optimal_enc['frontier_dist'] = 0.0

# Drop rows with NaNs in relevant columns
optimal_enc.dropna(subset=['accuracy', 'complexity', 'qc'], inplace=True)
shuffled_enc.dropna(subset=['accuracy', 'complexity', 'qc', 'frontier_dist'], inplace=True)
nl_enc.dropna(subset=['accuracy', 'complexity', 'qc', 'frontier_dist'], inplace=True)
nl_shuffled_enc.dropna(subset=['accuracy', 'complexity', 'qc', 'frontier_dist'], inplace=True)


########################################################################################################################
# Plot "the big plot" with the frontier points in a line and unique identifiers for other points

cmap = colormaps.get_cmap('viridis')

# Get optimal points
x_opt = optimal_enc['complexity']
y_opt = optimal_enc['accuracy']
col_opt = optimal_enc['qc']
col_fix = col_opt.min()
col_fix = cmap(col_fix)
# Get NL points
x_nl = nl_enc['complexity']
y_nl = nl_enc['accuracy']
col_nl = nl_enc['qc']
# Get shuffled suboptimal points
x_so = shuffled_enc['complexity']
y_so = shuffled_enc['accuracy']
col_so = shuffled_enc['qc']
# Get shuffled NL points
x_snl = nl_shuffled_enc['complexity']
y_snl = nl_shuffled_enc['accuracy']
col_snl = nl_shuffled_enc['qc']

# TODO: test and possibly remove this
#all_qc = np.concatenate((col_nl, col_so, col_snl))
all_qc = np.concatenate((col_nl, col_so))
min_qc = min(all_qc) * 0.9

# Create Chart
fig = plt.figure()
ax1 = fig.add_subplot(111)
# Populate Chart
ax1.scatter(x_opt, y_opt, s=10, c=col_opt, marker="o", cmap='viridis', vmin=0.0, vmax=1.0)
cb = ax1.scatter(x_so, y_so, s=10, c=col_so, marker="o", label='artificial encoders', cmap='viridis', vmin=min_qc, vmax=1.0)
ax1.scatter(x_snl,y_snl, s=10, c=col_snl, marker="o", cmap='viridis', vmin=min_qc, vmax=1.0)
ax1.scatter(x_nl,y_nl, s=30, c='gray', marker="s", cmap='viridis', vmin=min_qc, vmax=1.0)
ax1.scatter(x_nl,y_nl, s=10, c=col_nl, marker="x", label='NL encoders', cmap='viridis', vmin=min_qc, vmax=1.0)
ax1.plot(x_opt, y_opt, c='gray')
plt.legend(loc='upper left')
fig.colorbar(cb, ax=ax1, label='QC')
# Add axis labels
plt.xlabel("Complexity")
plt.ylabel("Accuracy")
# Add title to the plot (optional)
plt.title("IB Frontier with Degree of Quasi-Concavity Indicated")
#plt.show()

# Save chart
chart1_path = f'{chart_dir}/all_encoders_chart2.png'
plt.savefig(chart1_path)

########################################################################################################################
# Zoom in on the big plot, looking at just the encoders from the natural languages

cmap = colormaps.get_cmap('viridis')

# Get NL points
x_nl = nl_enc['complexity']
y_nl = nl_enc['accuracy']
col_nl = nl_enc['qc']

# Get limits of NL points
y_max = max(y_nl) * 1.1
x_max = max(x_nl) * 1.1
x_min = min(x_nl) * 0.9
y_min = min(y_nl) * 0.9

# Get optimal points
zoomed_opt_enc = optimal_enc[optimal_enc['complexity'] <= x_max]
zoomed_opt_enc = zoomed_opt_enc[zoomed_opt_enc['complexity'] >= x_min]
zoomed_opt_enc = zoomed_opt_enc[zoomed_opt_enc['accuracy'] <= y_max]
zoomed_opt_enc = zoomed_opt_enc[zoomed_opt_enc['accuracy'] >= y_min]
x_opt = zoomed_opt_enc['complexity']
y_opt = zoomed_opt_enc['accuracy']
col_opt = zoomed_opt_enc['qc']

# Get shuffled encoders within the range
zoomed_shuffled_enc = shuffled_enc[shuffled_enc['complexity'] <= x_max]
zoomed_shuffled_enc = zoomed_shuffled_enc[zoomed_shuffled_enc['complexity'] >= x_min]
zoomed_shuffled_enc = zoomed_shuffled_enc[zoomed_shuffled_enc['accuracy'] <= y_max]
zoomed_shuffled_enc = zoomed_shuffled_enc[zoomed_shuffled_enc['accuracy'] >= y_min]
x_shuff = zoomed_shuffled_enc['complexity']
y_shuff = zoomed_shuffled_enc['accuracy']

# Get shuffled NL encoders within the range
zoomed_snl_enc = nl_shuffled_enc[nl_shuffled_enc['complexity'] <= x_max]
zoomed_snl_enc = zoomed_snl_enc[zoomed_snl_enc['complexity'] >= x_min]
zoomed_snl_enc = zoomed_snl_enc[zoomed_snl_enc['accuracy'] <= y_max]
zoomed_snl_enc = zoomed_snl_enc[zoomed_snl_enc['accuracy'] >= y_min]
x_snl = zoomed_snl_enc['complexity']
y_snl = zoomed_snl_enc['accuracy']

# Create Chart
fig = plt.figure()
ax1 = fig.add_subplot(111)
# Populate Chart
ax1.scatter(x_shuff, y_shuff, s=10, c='gray', marker="o", label='shuffled encoders')
ax1.scatter(x_snl, y_snl, s=10, c='gray', marker="o")
ax1.plot(x_opt, y_opt, c='dimgray')
ax1.scatter(x_nl, y_nl, s=15, c=col_nl, marker="x", label='NL encoders', cmap='viridis', vmin=0.0, vmax=1.0)
plt.xlabel('Complexity')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
# Add axis labels
plt.xlabel("Complexity")
plt.ylabel("Accuracy")
# Add title to the plot (optional)
plt.title("IB Frontier with Degree of Quasi-Concavity for Natural Languages")
#plt.show()

# Save chart
chart2_path = f'{chart_dir}/nl_encoders_chart.png'
plt.savefig(chart2_path)

########################################################################################################################
# Get the relative importance of accuracy, complexity and QC against frontier distance for each set of points

# Indicate whether an intercept should be included or not
fit_int = True

# NL Encoders
xvals1 = nl_enc[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
yvals1 = nl_enc['frontier_dist'].to_numpy(copy=True)
nl_reg = LinearRegression(fit_intercept=fit_int)
nl_reg.fit(xvals1, yvals1)

# Suboptimal Shuffled Encoders
xvals2 = shuffled_enc[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
yvals2 = shuffled_enc['frontier_dist'].to_numpy(copy=True)
shuff_reg = LinearRegression(fit_intercept=fit_int)
shuff_reg.fit(xvals2, yvals2)

# NL Shuffled Encoders
xvals3 = nl_shuffled_enc[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
yvals3 = nl_shuffled_enc['frontier_dist'].to_numpy(copy=True)
nls_reg = LinearRegression(fit_intercept=fit_int)
nls_reg.fit(xvals3, yvals3)

# All
# Get optimal encoders
xvals0 = optimal_enc[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
yvals0 = optimal_enc['frontier_dist'].to_numpy(copy=True)
xvals4 = np.concatenate((xvals0, xvals1, xvals2, xvals3))
yvals4 = np.concatenate((yvals0, yvals1, yvals2, yvals3))
all_reg = LinearRegression(fit_intercept=fit_int)
all_reg.fit(xvals4, yvals4)

neg_reg = LinearRegression(fit_intercept=fit_int)
neg_reg.fit(xvals4, (-1.0*yvals4))

# Print results
print(f'Encoder \t QC \t Acc \t Comp')
print(f'NL \t {nl_reg.coef_[0]} \t {nl_reg.coef_[1]} \t {nl_reg.coef_[2]}')
print(f'Shuff \t {shuff_reg.coef_[0]} \t {shuff_reg.coef_[1]} \t {shuff_reg.coef_[2]}')
print(f'NL Shuff \t {nls_reg.coef_[0]} \t {nls_reg.coef_[1]} \t {nls_reg.coef_[2]}')
print(f'All \t {all_reg.coef_[0]} \t {all_reg.coef_[1]} \t {all_reg.coef_[2]}')
print(f'Negative All \t {neg_reg.coef_[0]} \t {neg_reg.coef_[1]} \t {neg_reg.coef_[2]}')

########################################################################################################################
# Get the correlations of accuracy, complexity, QC, and frontier distance for each set of points

# Natural languages
nl_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']].corr()

# Suboptimal shuffled encoders
shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']].corr()

# NL shuffled encoders
nl_shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']].corr()

# All
all_df = pd.concat((
    optimal_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    nl_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    nl_shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']]
))
print(all_df.corr())


########################################################################################################################
# Get the R2 and R2Diff values of accuracy, complexity, QC, and frontier distance for each set of points
# R2Diff = R2All - R2MissingVariableOfInterest

# Indicate whether to include an intercept
fit_int = True

all_df = pd.concat((
    optimal_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    nl_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    nl_shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']]
))

# all_df = pd.concat((
#     optimal_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
#     nl_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
#     shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']]
# ))

# R2 All
xvals_all = all_df[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
yvals = all_df['frontier_dist'].to_numpy(copy=True)
all_reg = LinearRegression(fit_intercept=fit_int)
all_reg.fit(xvals_all, yvals)
y_pred_all = all_reg.predict(xvals_all)
R2_all = r2_score(yvals, y_pred_all)

# R2 QC
xvals_qc = all_df['qc'].to_numpy(copy=True).reshape(-1, 1)
qc_reg = LinearRegression(fit_intercept=fit_int)
qc_reg.fit(xvals_qc, yvals)
y_pred_qc = qc_reg.predict(xvals_qc)
R2_qc = r2_score(yvals, y_pred_qc)

# R2 Accuracy
xvals_acc = all_df['accuracy'].to_numpy(copy=True).reshape(-1, 1)
acc_reg = LinearRegression(fit_intercept=fit_int)
acc_reg.fit(xvals_acc, yvals)
y_pred_acc = acc_reg.predict(xvals_acc)
R2_acc = r2_score(yvals, y_pred_acc)

# R2 Complexity
xvals_comp = all_df['complexity'].to_numpy(copy=True).reshape(-1, 1)
comp_reg = LinearRegression(fit_intercept=fit_int)
comp_reg.fit(xvals_comp, yvals)
y_pred_comp = comp_reg.predict(xvals_comp)
R2_comp = r2_score(yvals, y_pred_comp)

# R2 Diff QC
xvals_no_qc = all_df[['accuracy', 'complexity']].to_numpy(copy=True)
no_qc_reg = LinearRegression(fit_intercept=fit_int)
no_qc_reg.fit(xvals_no_qc, yvals)
y_pred_no_qc = no_qc_reg.predict(xvals_no_qc)
R2_no_qc = r2_score(yvals, y_pred_no_qc)
R2_diff_qc = R2_all - R2_no_qc

# R2 Diff Accuracy
xvals_no_acc = all_df[['qc', 'complexity']].to_numpy(copy=True)
no_acc_reg = LinearRegression(fit_intercept=fit_int)
no_acc_reg.fit(xvals_no_acc, yvals)
y_pred_no_acc = no_acc_reg.predict(xvals_no_acc)
R2_no_acc = r2_score(yvals, y_pred_no_acc)
R2_diff_acc = R2_all - R2_no_acc

# R2 Complexity
xvals_no_comp = all_df[['qc', 'accuracy']].to_numpy(copy=True)
no_comp_reg = LinearRegression(fit_intercept=fit_int)
no_comp_reg.fit(xvals_no_comp, yvals)
y_pred_no_comp = no_comp_reg.predict(xvals_no_comp)
R2_no_comp = r2_score(yvals, y_pred_no_comp)
R2_diff_comp = R2_all - R2_no_comp

# R2 Diff Accuracy and Complexity
R2_diff_acc_comp = R2_all - R2_qc

# Print out the results
print('R2 Values:')
print('QC \t Accuracy \t Complexity')
print(f'{R2_qc} \t {R2_acc} \t {R2_comp}')
print('')
print('R2 Diff Values:')
print('QC \t Accuracy \t Complexity \t Acc&Comp')
print(f'{R2_diff_qc} \t {R2_diff_acc} \t {R2_diff_comp} \t{R2_diff_acc_comp}')


########################################################################################################################
# Get the R2 and R2Diff values of accuracy, complexity, QC, and frontier distance for the natural languages

# Indicate whether to include an intercept
fit_int = True

# R2 All
xvals_all = nl_enc[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
yvals = nl_enc['frontier_dist'].to_numpy(copy=True)
all_reg = LinearRegression(fit_intercept=fit_int)
all_reg.fit(xvals_all, yvals)
y_pred_all = all_reg.predict(xvals_all)
R2_all = r2_score(yvals, y_pred_all)

# R2 QC
xvals_qc = nl_enc['qc'].to_numpy(copy=True).reshape(-1, 1)
qc_reg = LinearRegression(fit_intercept=fit_int)
qc_reg.fit(xvals_qc, yvals)
y_pred_qc = qc_reg.predict(xvals_qc)
R2_qc = r2_score(yvals, y_pred_qc)

# R2 Accuracy
xvals_acc = nl_enc['accuracy'].to_numpy(copy=True).reshape(-1, 1)
acc_reg = LinearRegression(fit_intercept=fit_int)
acc_reg.fit(xvals_acc, yvals)
y_pred_acc = acc_reg.predict(xvals_acc)
R2_acc = r2_score(yvals, y_pred_acc)

# R2 Complexity
xvals_comp = nl_enc['complexity'].to_numpy(copy=True).reshape(-1, 1)
comp_reg = LinearRegression(fit_intercept=fit_int)
comp_reg.fit(xvals_comp, yvals)
y_pred_comp = comp_reg.predict(xvals_comp)
R2_comp = r2_score(yvals, y_pred_comp)

# R2 Diff QC
xvals_no_qc = nl_enc[['accuracy', 'complexity']].to_numpy(copy=True)
no_qc_reg = LinearRegression(fit_intercept=fit_int)
no_qc_reg.fit(xvals_no_qc, yvals)
y_pred_no_qc = no_qc_reg.predict(xvals_no_qc)
R2_no_qc = r2_score(yvals, y_pred_no_qc)
R2_diff_qc = R2_all - R2_no_qc

# R2 Diff Accuracy
xvals_no_acc = nl_enc[['qc', 'complexity']].to_numpy(copy=True)
no_acc_reg = LinearRegression(fit_intercept=fit_int)
no_acc_reg.fit(xvals_no_acc, yvals)
y_pred_no_acc = no_acc_reg.predict(xvals_no_acc)
R2_no_acc = r2_score(yvals, y_pred_no_acc)
R2_diff_acc = R2_all - R2_no_acc

# R2 Diff Complexity
xvals_no_comp = nl_enc[['qc', 'accuracy']].to_numpy(copy=True)
no_comp_reg = LinearRegression(fit_intercept=fit_int)
no_comp_reg.fit(xvals_no_comp, yvals)
y_pred_no_comp = no_comp_reg.predict(xvals_no_comp)
R2_no_comp = r2_score(yvals, y_pred_no_comp)
R2_diff_comp = R2_all - R2_no_comp

# R2 Diff Accuracy and Complexity
R2_diff_comp = R2_all - R2_qc

# Print out the results
print('NL R2 Values:')
print('QC \t Accuracy \t Complexity')
print(f'{R2_qc} \t {R2_acc} \t {R2_comp}')
print('')
print('NL R2 Diff Values:')
print('QC \t Accuracy \t Complexity')
print(f'{R2_diff_qc} \t {R2_diff_acc} \t {R2_diff_comp}')

########################################################################################################################
# Get violin plots of QC values for each encoder type

# Optimal encoders
opt_qc = optimal_enc['qc'].values
opt_qc = {
    'qc': opt_qc,
    'type': ['ib optimal encoders'] * opt_qc.shape[0]
}
opt_df = pd.DataFrame(opt_qc)

# Natural language encoders
nl_qc = nl_enc['qc'].values
nl_qc = {
    'qc': nl_qc,
    'type': ['natural langauge encoders'] * nl_qc.shape[0]
}
nl_df = pd.DataFrame(nl_qc)

# Synthetic shuffled encoders
shuff_qc = shuffled_enc['qc'].values
shuff_qc = {
    'qc': shuff_qc,
    'type': ['shuffled synthetic encoders'] * shuff_qc.shape[0]
}
shuff_df = pd.DataFrame(shuff_qc)

# NL shuffled encoders
snl_qc = nl_shuffled_enc['qc'].values
snl_qc = {
    'qc': snl_qc,
    'type': ['shuffled NL encoders'] * snl_qc.shape[0]
}
snl_df = pd.DataFrame(snl_qc)

# Define the whole dataset
all_qc_df = pd.concat((opt_df, nl_df, shuff_df, snl_df))
enc_types = [x for x in all_qc_df['type'].unique()]

from plotnine import ggplot, aes, geom_violin, geom_boxplot, scale_x_discrete, coord_flip

# Make violin plots
violin_plt = (
        ggplot(all_qc_df, aes("type", "qc"))
        + geom_violin(all_qc_df)
        + coord_flip()
)
violin_plt.save(f'{chart_dir}/violin_plots.png')

# Make box plots
box_plt = (
    ggplot(all_qc_df)
    + geom_boxplot(aes(x="factor(type)", y="qc"))
    + coord_flip()
    + scale_x_discrete(labels=enc_types, name="Encoder Type")  # change ticks labels on OX
)
box_plt.save(f'{chart_dir}/box_plts.png')


########################################################################################################################
# Get violin plots of QC values for each encoder type with shuffled encoders combined

# Optimal encoders
opt_qc = optimal_enc['qc'].values
opt_qc = {
    'qc': opt_qc,
    'type': ['ib optimal encoders'] * opt_qc.shape[0]
}
opt_df = pd.DataFrame(opt_qc)

# Natural language encoders
nl_qc = nl_enc['qc'].values
nl_qc = {
    'qc': nl_qc,
    'type': ['natural langauge encoders'] * nl_qc.shape[0]
}
nl_df = pd.DataFrame(nl_qc)

# Synthetic shuffled encoders
shuff_qc1 = shuffled_enc['qc'].values
shuff_qc2 = nl_shuffled_enc['qc'].values
shuff_qc = np.concatenate((shuff_qc1, shuff_qc2))
shuff_qc = {
    'qc': shuff_qc,
    'type': ['shuffled encoders'] * shuff_qc.shape[0]
}
shuff_df = pd.DataFrame(shuff_qc)

# Define the whole dataset
all_qc_df = pd.concat((opt_df, nl_df, shuff_df))
enc_types = [x for x in all_qc_df['type'].unique()]

from plotnine import ggplot, aes, geom_violin, geom_boxplot, scale_x_discrete, coord_flip

# Make violin plots
violin_plt = (
        ggplot(all_qc_df, aes("type", "qc"))
        + geom_violin(all_qc_df)
        + coord_flip()
)
violin_plt.save(f'{chart_dir}/violin_plots2.png')

# Make box plots
box_plt = (
    ggplot(all_qc_df)
    + geom_boxplot(aes(x="factor(type)", y="qc"))
    + coord_flip()
    + scale_x_discrete(labels=enc_types, name="Encoder Type")  # change ticks labels on OX
)
box_plt.save(f'{chart_dir}/box_plts2.png')

########################################################################################################################
# Get the Linear Regression coefficients with statistical significance

# Indicate whether to include an intercept
fit_int = True

all_df = pd.concat((
    optimal_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    nl_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']],
    nl_shuffled_enc[['qc', 'accuracy', 'complexity', 'frontier_dist']]
))

# Fit Linear Regression
X = all_df[['qc', 'accuracy', 'complexity']].to_numpy(copy=True)
Y = all_df['frontier_dist'].to_numpy(copy=True)
all_reg = LinearRegression(fit_intercept=fit_int)
all_reg.fit(X, Y)
#y_pred_all = all_reg.predict(xvals_all)

# Get statistics
import statsmodels.api as sm
X = sm.add_constant(X)
model_sm = sm.OLS(Y, X)
results = model_sm.fit()
print(results.summary())
