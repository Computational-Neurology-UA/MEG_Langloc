from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import scipy.stats
import math
import mne
import os


# Parameter settings
language = 'dutch'

if language == 'english':
    all_subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
else:
    all_subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20]

picks = 'grad'
num_sensors = 204 if picks=='grad' else 102
    
base = (-0.05, 0.05)
fs = 500
language_window = (0.1, 0.35)
language_samples = (int(np.rint((language_window[0]-base[0])*fs)), 
                    int(np.rint((language_window[1]-base[0])*fs)))
main_path = './data/spatiotemporal'

def convert_pvalue_to_asterisks(pvalue, only_asterisk=True):
    if pvalue <= 0.0001:
        return '****' if only_asterisk else f"**** {pvalue:.2e}"
    elif pvalue <= 0.001:
        return '***' if only_asterisk else f"*** {pvalue:.2e}"
    elif pvalue <= 0.01:
        return '**' if only_asterisk else f"** {pvalue:.2e}"
    elif pvalue <= 0.05:
        return '*' if only_asterisk else f"* {pvalue:.2e}"
    return "ns"

def percentage_formatter(x, pos):
    return f"{x:.0f}%"

formatter = FuncFormatter(percentage_formatter)

# Create dummy analysis to extract info object
epoch_psd = mne.read_epochs('data/preproc/dutch/sensor_data/funcloc_preproc_s1-epo.fif').compute_psd(fmin=1, fmax=40)
info = mne.pick_info(epoch_psd.info, sel=mne.pick_types(epoch_psd.info, meg='grad'))

#%% Splithalf reliability 1: Within vs between groups correlation (first vs others)

# Get saving structure
within_corr = []
between_corr = []

for subject_id in all_subjects:
    
    fpath = os.path.join(main_path, language, f's{subject_id}')
    fname_cluster = f'cluster_stats_{picks}.npy'
    cluster_stats_splithalf = np.load(os.path.join(fpath, fname_cluster), allow_pickle=True)[()]
    
    # Compute correlation value in fmaps
    f_first, _, _, _ = cluster_stats_splithalf['first']
    f_last, _, _, _ = cluster_stats_splithalf['last']
    
    # Select relevant language time window
    f_first = np.mean(f_first[language_samples[0]:language_samples[1], :], axis=0)
    f_last = np.mean(f_last[language_samples[0]:language_samples[1], :], axis=0)
    
    # Compute within correlation
    res = scipy.stats.spearmanr(f_first, f_last)
    within_corr.append(res.statistic)
    
    for other_id in all_subjects:
        
        if subject_id != other_id:
            fpath = os.path.join(main_path, language, f's{other_id}')
            fname_cluster = f'cluster_stats_{picks}.npy'
            cluster_stats_splithalf = np.load(os.path.join(fpath, fname_cluster), allow_pickle=True)[()]
            
            # Compute correlation value in fmaps
            f_last_other, _, _, _ = cluster_stats_splithalf['last']
            
            # Select relevant language time window
            f_last_other = np.mean(f_last_other[language_samples[0]:language_samples[1], :], axis=0)
            
            # Compute within correlation
            res = scipy.stats.spearmanr(f_first, f_last_other)
            between_corr.append(res.statistic)

mu_within = np.mean(within_corr)
mu_between = np.mean(between_corr)

print(f'Within corr avg rho: {mu_within}')
print(f'Within corr std rho: {np.std(within_corr)}')
print(f'Between corr avg rho: {mu_between}')
print(f'Between corr std rho: {np.std(between_corr)}')

if language == 'dutch':
    color_within = 'steelblue'
    color_between = 'lightsteelblue'
else:
    color_within = 'orchid'
    color_between = 'thistle'

bins = np.arange(-0.75, 1, 0.05)

fig, ax = plt.subplots(1, 1, figsize=(5,4))

hist_between, bins_between, p_between = ax.hist(between_corr, bins=bins, density=True, color=color_between, label='Between')
hist_within, bins_within, p_within = ax.hist(within_corr, bins=bins, density=True, color=color_within, label='Within')

for item_between, item_within in zip(p_between, p_within):
    item_between.set_height(item_between.get_height()/max(hist_between))
    item_within.set_height(item_within.get_height()/max(hist_within))

ax.set_xlabel('Spearman correlation')
ax.set_ylim([0, 1.19])
ax.axvline(x=mu_within, ymax=1.05/1.19, color='darkslategray', linestyle='--', linewidth=1.2)
ax.axvline(x=mu_between, ymax=1.05/1.19, color='slategray', linestyle='--', linewidth=1.2)
ax.legend()
ax.set_ylabel('Relative frequency')

# Statistical test with Wilcoxon Rank Sum test
stat, pval = scipy.stats.ranksums(np.ravel(within_corr), between_corr)
print(f'Test statistic: {stat}')
print(f'Pvalue: {pval}')
print('=============')
corr_low, corr_high = mu_between, mu_within
min_val, max_val = 1.05, 1.10
ax.plot([corr_low, corr_low, corr_high, corr_high], [min_val, max_val, max_val, min_val], lw=1.5, c='k')
ax.text((corr_low + corr_high)*.5, max_val, convert_pvalue_to_asterisks(pval) , ha='center', va='bottom', c='k')
sns.despine(trim=True)  # Remove top and right spines for a cleaner look


#%% Splithalf reliability 2a: similarity in topoplots

# Get SOIs as highest 10% sensors
soi = {'first':dict.fromkeys(np.arange(0, len(all_subjects))), 
       'last':dict.fromkeys(np.arange(0, len(all_subjects)))}

soi_weight = {'first':dict.fromkeys(np.arange(0, len(all_subjects))), 
              'last':dict.fromkeys(np.arange(0, len(all_subjects)))}

# Loop over all participants
for subject_idx, subject_id in enumerate(all_subjects):
    
    fpath = os.path.join(main_path, language, f's{subject_id}')
    fname_cluster = f'cluster_stats_{picks}.npy'
    cluster_stats_splithalf = np.load(os.path.join(fpath, fname_cluster), allow_pickle=True)[()]
    
    # Compute correlation value in fmaps
    f_first, _, _, _ = cluster_stats_splithalf['first']
    f_last, _, _, _ = cluster_stats_splithalf['last']
    
    # Select relevant language time window
    f_first = np.mean(f_first[language_samples[0]:language_samples[1], :], axis=0)
    f_last = np.mean(f_last[language_samples[0]:language_samples[1], :], axis=0)
    
    for split, fmap in zip(['first', 'last'], [f_first, f_last]):
        
        fmap_copy = np.copy(fmap)
        fmap_copy.sort()
        
        sensors = fmap_copy[len(fmap_copy) - math.ceil(len(fmap_copy)/10):]
        sensors = fmap_copy[204 - 10:]
        idx_sensor = [idx for idx, val in enumerate(fmap) if val in sensors]
        
        soi[split][subject_idx] = idx_sensor
        soi_weight[split][subject_idx] = fmap
    

# Create dummy analysis to extract info object
subject_idx = 8

# Plot SOIs
fig, axes = plt.subplots(2, 1, figsize=(4, 8))

for ax, split in zip(axes.flatten(), ['first', 'last']):
    
    mask = [False for i in range(102)]
    for soi_idx in soi[split][subject_idx]:
        mask[soi_idx//2] = True
    mask = np.array(mask)
    
    data = soi_weight[split][subject_idx]
    
    im, cm = mne.viz.plot_topomap(data, info, mask=mask, axes=ax,
                                  cmap='viridis', vlim=(0, 8), 
                                  mask_params=(dict(marker='o', markersize=5.5)))
    
    # Get the position of the topomap axis in figure coordinates
    pos = ax.get_position()
    
    # Create a divider with fixed-size offset (pad, width) in inches
    pad_inches = 0.1
    width_inches = 0.18
    divider = make_axes_locatable(ax)
    
    # Create a new axis manually that matches the topomap’s height
    cax = fig.add_axes([
            pos.x1 + pad_inches / fig.get_size_inches()[0], 
            pos.y0 + 0.01,                                          
            width_inches / fig.get_size_inches()[0],   
            pos.height -0.05])
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    
    cbar.ax.tick_params(labelsize=20)  # increase tick font size
    cbar.ax.yaxis.label.set_family('Roboto')  # set font family
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontname('Roboto')


#%% Splithalf reliability 2b: similarity in topoplots

# Get SOIs as highest 10% sensors
fmaps = {'first': [], 'last': []}
soi_group = {'first':None, 'last':None}
soi_group_weight = {'first':None, 'last':None}

# Loop over all participants
for subject_idx, subject_id in enumerate(all_subjects):
    
    fpath = os.path.join(main_path, language, f's{subject_id}')
    fname_cluster = f'cluster_stats_{picks}.npy'
    cluster_stats_splithalf = np.load(os.path.join(fpath, fname_cluster), allow_pickle=True)[()]
    
    # Compute correlation value in fmaps
    f_first, _, _, _ = cluster_stats_splithalf['first']
    f_last, _, _, _ = cluster_stats_splithalf['last']
    
    # Select relevant language time window
    f_first = np.mean(f_first[language_samples[0]:language_samples[1], :], axis=0)
    f_last = np.mean(f_last[language_samples[0]:language_samples[1], :], axis=0)
    
    # Save fmaps
    fmaps['first'].append(f_first)
    fmaps['last'].append(f_last)
    

for split in ['first', 'last']:
    
    # Compute group average of fmaps
    f_groupavg = np.mean(fmaps[split], axis=0)
    
    # Compute SOIs
    fmap_copy = np.copy(f_groupavg)
    fmap_copy.sort()
    
    sensors = fmap_copy[len(fmap_copy) - math.ceil(len(fmap_copy)/10):]
    idx_sensor = [idx for idx, val in enumerate(f_groupavg) if val in sensors]
    
    soi_group[split] = idx_sensor
    soi_group_weight[split] = f_groupavg
    
# Plot SOIs
fig, axes = plt.subplots(2, 1, figsize=(4, 8))

for ax, split in zip(axes.flatten(), ['first', 'last']):
    
    mask = [False for i in range(102)]
    for soi_idx in soi_group[split]:
        mask[soi_idx//2] = True
    mask = np.array(mask)
    
    data = soi_group_weight[split]
    
    im, cm = mne.viz.plot_topomap(data, info, mask=mask, axes=ax, 
                                  cmap='viridis', vlim=(0, 4),
                                  mask_params=(dict(marker='o', markersize=5.5)))
     
    # Get the position of the topomap axis in figure coordinates
    pos = ax.get_position()
    
    # Create a divider with fixed-size offset (pad, width) in inches
    pad_inches = 0.1
    width_inches = 0.18
    divider = make_axes_locatable(ax)
    
    # Create a new axis manually that matches the topomap’s height
    cax = fig.add_axes([
            pos.x1 + pad_inches / fig.get_size_inches()[0], 
            pos.y0 + 0.01,                                          
            width_inches / fig.get_size_inches()[0],   
            pos.height -0.05])
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    

#%% Probability overlap distribution map

soi_counts = {'first': [], 'last': []}

for split in ['first', 'last']:
    
    soi_counter = [0 for i in range(204)]
    
    for subject_idx in range(len(all_subjects)):
        soi_subject_data = soi[split][subject_idx]
        
        for soi_present in soi_subject_data:
            soi_counter[soi_present] += 1
            
    soi_counts[split] = soi_counter

# Plot probability overlap map
fig, axes = plt.subplots(2, 1, figsize=(5, 8), tight_layout=True)

for ax, split in zip(axes.flatten(), ['first', 'last']):
    
    # Convert the counts to percentages
    conversion_factor = 100/len(all_subjects)
    data = [i*conversion_factor for i in soi_counts[split]]
    print(f'Most present SOI is seen in {max(data)}% of participants')
    
    mask = [False for i in range(102)]
    for soi_idx in soi_group[split]:
        mask[soi_idx//2] = True
    mask = np.array(mask)
    
    im, _ = mne.viz.plot_topomap(data, info, axes=ax, cmap='viridis', vlim=(0, 100),
        mask=mask, mask_params=dict(marker='o', markersize=4.5))
    
    ax.set_title(f"Probability overlap map - {split} split")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', label='Probability [%]')
    cbar.ax.yaxis.set_major_formatter(formatter)

#%% Splithalf reliability figure 3: Effect size comparison

effect_subject = []
effect_group = []

soi_selection = 'first'
soi_computation = 'last'

# With individually defined SOI
for subject_idx in range(len(all_subjects)):

    # Fetch relevant data
    f_last = fmaps[soi_computation][subject_idx]
    soi_subject_level = soi[soi_selection][subject_idx]
    soi_group_level = soi_group[soi_selection]

    # Take avg of fmap in soi
    f_subject_level = np.mean(f_last[soi_subject_level])
    f_group_level = np.mean(f_last[soi_group_level])

    # Save data
    effect_subject.append(f_subject_level)
    effect_group.append(f_group_level)

# Visualize all results with barplot + swarm + connecting lines
fig, ax = plt.subplots(1, 1, figsize=(4, 5), tight_layout=True)

# Prepare data
data = [effect_subject, effect_group]
labels = ['Ind. SOI', 'Group SOI']

# Effect size increase
data_diff = [data[0][idx] - data[1][idx] for idx in range(len(all_subjects))]
print(f'Mean effect size ind. SOIs: {np.mean(effect_subject)}')
print(f'Mean effect size group SOIs: {np.mean(effect_group)}')
print(f'Mean diff in effect size: {np.mean(data_diff)}')

# Colors for the barplots / points
colors = ['steelblue', 'lightsteelblue'] if language == 'dutch' else ['purple', 'orchid']

# Create barplot (mean ± SD)
bar = ax.bar(
    ['Ind. SOI', 'Group SOI'],
    [np.mean(effect_subject), np.mean(effect_group)],
    yerr=[np.std(effect_subject), np.std(effect_group)],
    edgecolor=colors,
    color=[0, 0, 0, 0],     # transparent fill
    linewidth=2,
    capsize=4)

# Add swarm (jittered points) + store x positions for line connections
x_positions = []

for idx, (color, datapoints) in enumerate(zip(colors, (effect_subject, effect_group))):

    # Get bounds around the bar center
    low_bound = bar.patches[idx].get_center()[0] - 0.2
    upper_bound = bar.patches[idx].get_center()[0] + 0.2

    # Random x values between bounds (one per subject)
    x_vals = np.random.uniform(low_bound, upper_bound, len(all_subjects))
    x_positions.append(x_vals)

    ax.scatter(x_vals, datapoints, c=color, linewidth=0.05, alpha=0.9, zorder=3)

# Add connecting lines between paired observations (Ind -> Group)
for x0, x1, y0, y1 in zip(x_positions[0], x_positions[1], effect_subject, effect_group):
    line = mlines.Line2D([x0 + 0.02, x1 - 0.02], [y0, y1],
                         color='lightgrey', alpha=0.8, zorder=2)
    ax.add_line(line)

# X tick formatting (keep your style)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, fontsize=10)


# Check normality
data = np.array(effect_subject) - np.array(effect_group)
stat_normality, p_value_normality = scipy.stats.shapiro(data)

print(f"  Shapiro-Wilk statistic: {stat_normality:.4f}")
print(f"  p-value: {p_value_normality:.4f}")

if p_value_normality > 0.05:
    print("  Data looks normally distributed (fail to reject H0)")    
    stat_test, pval_test = scipy.stats.ttest_rel(effect_subject, effect_group,
                                                 alternative='greater')

else:
    print("  Data is not normally distributed (reject H0)")
    stat_test, pval_test = scipy.stats.wilcoxon(effect_subject, effect_group,
                                                alternative='greater')

# --- Q-Q plot ---
plt.figure()
scipy.stats.probplot(data, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()
    
print(f'Test statistic of individual vs group based SOIs: {stat_test}')
print(f'Pvalue of individual vs group based SOIs: {pval_test}')

# Define increases in individual SOI vs group level SOI
substraction_list = [effect_subject[i] - effect_group[i] for i in range(len(all_subjects))]
amount_greater = sum(x > 0 for x in substraction_list)
print(f'Amount of sent ind > sent group: {amount_greater}')

# Significance bracket + asterisks
min_val = max(max(effect_subject), max(effect_group)) * 1.1
max_val = max(max(effect_subject), max(effect_group)) * 1.15

ax.plot([0, 0, 1, 1], [min_val, max_val, max_val, min_val], lw=1.5, c='k')
ax.text(0.5, max_val, convert_pvalue_to_asterisks(pval_test),
        ha='center', va='bottom', c='k')

# Plot layout
ax.set_ylabel('F-values')
sns.despine(trim=True)

