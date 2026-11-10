from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import seaborn as sns
import mne
import os
import gc


# Setup paths
SUBJECTS_DIR = '/mnt/d/MEG_LangLoc/Dutch/MRI/Freesurfer'
SOURCE_SPACE_DIR = '/mnt/d/MEG_LangLoc/Dutch/source_space'
CONTRAST_MAP_DIR = './data/preproc/dutch/contrast_map'
MORPH_DIR = './data/preproc/dutch/morphs'
OUTPUT_DIR = './data/preproc/dutch/source_SOI_results'

use_gradonly = True
save_results = False

if use_gradonly:
    SOURCE_EST_DIR = '/mnt/d/MEG_LangLoc/Dutch/source_estimate_gradonly'
    CLUSTER_STATS_FNAME = 'cluster_stats_gradonly.npy'
else:
    SOURCE_EST_DIR = '/mnt/d/MEG_LangLoc/Dutch/source_estimate'
    CLUSTER_STATS_FNAME = 'cluster_stats_base.npy'

# Setup analysis params
SUBJECT_LIST = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                's11', 's12', 's13', 's15', 's16', 's17', 's18', 's19', 's20']
SPLITHALVES = ['first', 'last']
CV_DIRECTIONS = [('first', 'last'), ('last',  'first')]

# Time window for F-map averaging and SOI definition (seconds)
TMIN_SOI = 0.100
TMAX_SOI = 0.350
TMIN_STC = - 0.05
TSTEP_STC = 0.002

# Top 10% of sources = SOI, matching sensor-space analysis
SOI_PERCENTILE = 90

# Smoothing steps for surface morphing
MORPH_SMOOTH = 5


#%% Helper functions: morphing (with caching) and Fmaps

def get_fsaverage_src(subjects_dir):
    """
    Load fsaverage oct6 source space.
    If fsaverage is not yet in subjects_dir, fetch it from MNE's built-in
    data using mne.datasets.fetch_fsaverage, which downloads/copies the
    fsaverage subject into the given subjects_dir.
    """
    fsaverage_dir = os.path.join(subjects_dir, 'fsaverage')
    if not os.path.exists(fsaverage_dir):
        print("  fsaverage not found in subjects_dir — fetching from MNE...")
        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)

    path = os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-oct6-src.fif')
    if os.path.exists(path):
        return mne.read_source_spaces(path)

    print("  Building fsaverage oct6 source space...")
    
    src = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    src.save(path, overwrite=True)
    
    return src


def get_morph_to_fsaverage(subject, src_subject, src_fsaverage, subjects_dir, morph_dir, smooth):
    
    path = os.path.join(morph_dir, f'{subject}-to-fsaverage-morph.h5')
    
    if os.path.exists(path):
        return mne.read_source_morph(path)
    
    morph = mne.compute_source_morph(src=src_subject, subject_from=subject, 
                                     subject_to='fsaverage', src_to=src_fsaverage,
                                     subjects_dir=subjects_dir, smooth=smooth)
    
    morph.save(path, overwrite=True)
    
    return morph


def get_morph_from_fsaverage(subject, src_subject, src_fsaverage, subjects_dir, morph_dir, smooth):
    
    path = os.path.join(morph_dir, f'fsaverage-to-{subject}-morph.h5')
    
    if os.path.exists(path):
        return mne.read_source_morph(path)
    
    morph = mne.compute_source_morph(src=src_fsaverage, subject_from='fsaverage', 
                                     subject_to=subject, src_to=src_subject,
                                     subjects_dir=subjects_dir, smooth=smooth)
    
    morph.save(path, overwrite=True)
    
    return morph


def time_average_fmap(F_obs, tmin, tstep, twin_min, twin_max):
    """
    F_obs : (n_times, n_sources) — spatio_temporal_cluster_test convention
    Returns (n_sources,) — mean F across the time window
    """
    times = tmin + np.arange(F_obs.shape[0]) * tstep
    tmask = (times >= twin_min) & (times <= twin_max)
    return F_obs[tmask, :].mean(axis=0)

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


#%% STEP 1: Morph per-subject, per-half F-maps to fsaverage

print("=" * 70)
print("STEP 1: Morphing individual F-maps to fsaverage (both splithalves)")
print("=" * 70)

src_fsaverage = get_fsaverage_src(SUBJECTS_DIR)
fsaverage_vertices = [src_fsaverage[0]['vertno'], src_fsaverage[1]['vertno']]

fmaps_fsavg = {subj: {} for subj in SUBJECT_LIST}

for subject in SUBJECT_LIST:
    
    print(f"\n  {subject}")
    src_subject = mne.read_source_spaces(os.path.join(SOURCE_SPACE_DIR, subject, 
                                                      f'src_{subject}-oct6-src.fif'))
    
    vertices_subject = [src_subject[0]['vertno'], src_subject[1]['vertno']]
    morph_to = get_morph_to_fsaverage(subject, src_subject, src_fsaverage, 
                                      SUBJECTS_DIR, MORPH_DIR, MORPH_SMOOTH)
    
    cluster_stats = np.load(os.path.join(CONTRAST_MAP_DIR, subject, CLUSTER_STATS_FNAME), 
                            allow_pickle=True).item()

    for half in SPLITHALVES:
        F_obs = cluster_stats[half][0]   # (n_times, n_sources)

        stc_fmap = mne.SourceEstimate(data=F_obs.T,
                                      vertices=vertices_subject,
                                      tmin=TMIN_STC, tstep=TSTEP_STC,
                                      subject=subject)
        stc_fsavg = morph_to.apply(stc_fmap)
        fmaps_fsavg[subject][half] = stc_fsavg   # keep full STC

        del F_obs, stc_fmap, stc_fsavg
        gc.collect()

    del src_subject, morph_to, cluster_stats
    gc.collect()


#%% STEP 2: Per-splithalf group-average F-map and SOI mask in fsaverage

print("\n" + "=" * 70)
print("STEP 2: Computing per-splithalf group F-maps in fsaverage")
print("=" * 70)

group_fmap_fsavg     = {}
group_soi_mask_fsavg = {}

for half in SPLITHALVES:

    # Compute and save group fmap with temporal information
    stack = np.array([fmaps_fsavg[s][half].data for s in SUBJECT_LIST])
    group_fmap = stack.mean(axis=0)
    
    if save_results:
        np.save(os.path.join(OUTPUT_DIR, f'group_fmap_fsaverage_{half}_gradonly.npy'), group_fmap)
    
    # Time-average in fsaverage space
    group_fmap = time_average_fmap(group_fmap.T, TMIN_STC, TSTEP_STC, TMIN_SOI, TMAX_SOI)
    group_fmap_fsavg[half] = group_fmap

    threshold = np.percentile(group_fmap, SOI_PERCENTILE)
    group_soi_mask_fsavg[half] = group_fmap >= threshold

    print(f"  {half}: F range [{group_fmap.min():.3f}, {group_fmap.max():.3f}]"
          f"  |  SOI: {group_soi_mask_fsavg[half].sum()} sources "
          f"(threshold = {threshold:.3f})")


#%% STEP 3: Backproject per-splithalf group SOI masks to individual native space

print("\n" + "=" * 70)
print("STEP 3: Backprojecting per-splithalf group SOI masks to native spaces")
print("=" * 70)

# native_group_soi[half][subject] -> boolean (n_sources_native,)
native_group_soi = {half: {} for half in SPLITHALVES}

for half in SPLITHALVES:
    print(f"\n  {half} half:")

    n_soi_fsavg = group_soi_mask_fsavg[half].sum()
    stc_mask_fsavg = mne.SourceEstimate(data=group_soi_mask_fsavg[half].astype(float)[:, np.newaxis],
                                        vertices=fsaverage_vertices,
                                        tmin=0.0, tstep=1.0,
                                        subject='fsaverage')

    for subject in SUBJECT_LIST:
        src_subject = mne.read_source_spaces(os.path.join(SOURCE_SPACE_DIR, subject,
                                                          f'src_{subject}-oct6-src.fif'))
        
        morph_from = get_morph_from_fsaverage(subject, src_subject, src_fsaverage, 
                                              SUBJECTS_DIR, MORPH_DIR, MORPH_SMOOTH)
        
        stc_mask_native = morph_from.apply(stc_mask_fsavg)
        native_vals = stc_mask_native.data[:, 0]

        native_mask = native_vals > 0

        # Fallback: if too few vertices survive, use top-N to preserve SOI size
        if native_mask.sum() < 0.5 * n_soi_fsavg:
            print(f"    Warning ({subject}): {native_mask.sum()} vertices after "
                  f"thresholding (expected ~{n_soi_fsavg}). Using top-N.")
            topN_thresh = np.sort(native_vals)[-n_soi_fsavg]
            native_mask = native_vals >= topN_thresh

        native_group_soi[half][subject] = native_mask
        print(f"    {subject}: {native_mask.sum()} native sources")

        del src_subject, stc_mask_native, native_vals
        gc.collect()


#%% STEP 4: Cross-validated SOI evaluation (both CV directions)

print("\n" + "=" * 70)
print("STEP 4: Cross-validated SOI evaluation (both CV directions)")
print("=" * 70)

# summary_f_individual[subject] and summary_f_group[subject]:
# lists of length 2 (one entry per CV direction), averaged later
summary_f_individual = {subj: [] for subj in SUBJECT_LIST}
summary_f_group = {subj: [] for subj in SUBJECT_LIST}

for define_half, eval_half in CV_DIRECTIONS:
    print(f"\n  Direction: define on '{define_half}' → evaluate on '{eval_half}'")

    for subject in SUBJECT_LIST:

        # Individual SOI: top 10% of defining-half F-map in native space
        cluster_stats = np.load(os.path.join(CONTRAST_MAP_DIR, subject, CLUSTER_STATS_FNAME), 
                                  allow_pickle=True).item()
        
        F_avg_define = time_average_fmap(cluster_stats[define_half][0],
                                         TMIN_STC, TSTEP_STC, 
                                         TMIN_SOI, TMAX_SOI)
        
        threshold_indiv = np.percentile(F_avg_define, SOI_PERCENTILE)
        indiv_soi_mask = F_avg_define >= threshold_indiv
        
        # Group SOI: backprojected mask from the same defining half
        group_soi_mask = native_group_soi[define_half][subject]
        
        F_avg_eval = time_average_fmap(cluster_stats[eval_half][0],
                                       TMIN_STC, TSTEP_STC,
                                       TMIN_SOI, TMAX_SOI)

        # Summary F: mean F-value across SOI vertices
        f_indiv = F_avg_eval[indiv_soi_mask].mean()
        f_group = F_avg_eval[group_soi_mask].mean()

        summary_f_individual[subject].append(f_indiv)
        summary_f_group[subject].append(f_group)

        print(f"    {subject}  indiv={f_indiv:.4f}  group={f_group:.4f}")


#%% STEP 5: Wilcoxon signed-rank test

print("\n" + "=" * 70)
print("STEP 5: Statistical test: individual SOIs vs group SOIs")
print("=" * 70)

selection_option = 0 # 0 if define on first and evaluate on last (else 1)

summary_f_individual_selected = np.array([summary_f_individual[s][selection_option] for s in SUBJECT_LIST])
summary_f_group_selected = np.array([summary_f_group[s][selection_option] for s in SUBJECT_LIST])

# Check normality
data = summary_f_individual_selected - summary_f_group_selected
stat_normality, p_value_normality = scipy.stats.shapiro(data)

print(f"  Shapiro-Wilk statistic: {stat_normality:.4f}")
print(f"  p-value: {p_value_normality:.4f}")

if p_value_normality > 0.05:
    print("  Data looks normally distributed (fail to reject H0)")    
    stat_test, pval_test = scipy.stats.ttest_rel(summary_f_individual_selected, summary_f_group_selected,
                                                 alternative='greater')

else:
    print("  Data is not normally distributed (reject H0)")
    stat_test, pval_test = scipy.stats.wilcoxon(summary_f_individual_selected, summary_f_group_selected,
                                                alternative='greater')
    
# --- Q-Q plot ---
plt.figure()
scipy.stats.probplot(data, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

n_individual_higher = (summary_f_individual_selected > summary_f_group_selected).sum()
mean_diff           = (summary_f_individual_selected - summary_f_group_selected).mean()

print(f"\n  N subjects with higher F for individual SOIs: "
      f"{n_individual_higher}/{len(SUBJECT_LIST)}")
print(f"  Mean difference (individual - group): {mean_diff:.4f}")
print(f"  Test statistic: {stat_test:.4f}")
print(f"  p-value (one-sided, individual > group): {pval_test:.4f}")

print("\n  Per-subject summary:")
print(f"  {'Subject':<10} {'Indiv SOI F':>12} {'Group SOI F':>12} {'Diff':>8}")
print("  " + "-" * 46)
for i, subj in enumerate(SUBJECT_LIST):
    diff = summary_f_individual_selected[i] - summary_f_group_selected[i]
    print(f"  {subj:<10} {summary_f_individual_selected[i]:>12.4f} "
          f"{summary_f_group_selected[i]:>12.4f} {diff:>8.4f}")


#%% Visualize results

fig, ax = plt.subplots(1, 1, figsize=(4, 5), tight_layout=True)

# Prepare data
data = [summary_f_individual_selected, summary_f_group_selected]
labels = ['Ind. SOI', 'Group SOI']

# Effect size increase
data_diff = [data[0][idx] - data[1][idx] for idx in range(len(summary_f_individual))]

# Colors for the barplots / points
colors = ['steelblue', 'lightsteelblue'] 

# Create barplot (mean ± SD)
bar = ax.bar(
    ['Ind. SOI', 'Group SOI'],
    [np.mean(summary_f_individual_selected), np.mean(summary_f_group_selected)],
    yerr=[np.std(summary_f_individual_selected), np.std(summary_f_group_selected)],
    edgecolor=colors,
    color=[0, 0, 0, 0],
    linewidth=2,
    capsize=4)

# Add swarm (jittered points) + store x positions for line connections
x_positions = []

for idx, (color, datapoints) in enumerate(zip(colors, (summary_f_individual_selected, summary_f_group_selected))):

    # Get bounds around the bar center
    low_bound = bar.patches[idx].get_center()[0] - 0.2
    upper_bound = bar.patches[idx].get_center()[0] + 0.2

    # Random x values between bounds (one per subject)
    x_vals = np.random.uniform(low_bound, upper_bound, len(summary_f_individual))
    x_positions.append(x_vals)

    ax.scatter(x_vals, datapoints, c=color, linewidth=0.05, alpha=0.9, zorder=3)

# Add connecting lines between paired observations (Ind -> Group)
for x0, x1, y0, y1 in zip(x_positions[0], x_positions[1], summary_f_individual_selected, summary_f_group_selected):
    line = mlines.Line2D([x0 + 0.02, x1 - 0.02], [y0, y1],
                         color='lightgrey', alpha=0.8, zorder=2)
    ax.add_line(line)

# X tick formatting (keep your style)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, fontsize=10)

# Significance bracket + asterisks
min_val = max(max(summary_f_individual_selected), max(summary_f_group_selected)) * 1.1
max_val = max(max(summary_f_individual_selected), max(summary_f_group_selected)) * 1.15

ax.plot([0, 0, 1, 1], [min_val, max_val, max_val, min_val], lw=1.5, c='k')
ax.text(0.5, max_val, convert_pvalue_to_asterisks(pval_test),
        ha='center', va='bottom', c='k')

# Plot layout
ax.set_ylabel('F-values')
sns.despine(trim=True)

