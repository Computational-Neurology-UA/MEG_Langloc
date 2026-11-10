from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats
import mne
import os

# Parameter settings
SUBJECT_LIST = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                's11', 's12', 's13', 's15', 's16', 's17', 's18', 's19', 's20']

# Timing params
base = (-0.05, 0.05)
fs = 500
tmin = -0.05
tstep = 0.002
language_window = (0.100, 0.350)
language_samples = (int(np.rint((language_window[0]-base[0])*fs)), 
                    int(np.rint((language_window[1]-base[0])*fs)))

# Setup paths
SUBJECTS_DIR     = '/mnt/d/MEG_LangLoc/Dutch/MRI/Freesurfer'
SOURCE_SPACE_DIR = '/mnt/d/MEG_LangLoc/Dutch/source_space'
CONTRAST_MAP_DIR = './data/preproc/dutch/contrast_map'
MORPH_DIR        = './data/preproc/dutch/morphs'

# Select gradiometer only or full sensor stcs
use_gradonly = False


#%% Define helper functions
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

def get_stc_vertices_from_src(subject_id, main_path):
    """
    Extract stc.vertices structure directly from the subject's src.fif file.
   
    Returns
    -------
    vertices : list of two arrays [lh_vertices, rh_vertices]
    """
    fpath = os.path.join(main_path, f's{subject_id}')
    src = mne.read_source_spaces(os.path.join(fpath, f'src_s{subject_id}-oct6-src.fif'))
    
    lh_verts = src[0]['vertno']
    rh_verts = src[1]['vertno']
    
    return [lh_verts, rh_verts]


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


def get_morph_to_fsaverage(subject, src_subject, src_fsaverage, subjects_dir, morph_dir, smooth=5):
    
    path = os.path.join(morph_dir, f'{subject}-to-fsaverage-morph.h5')
    
    if os.path.exists(path):
        return mne.read_source_morph(path)
    
    morph = mne.compute_source_morph(src=src_subject, subject_from=subject, 
                                     subject_to='fsaverage', src_to=src_fsaverage,
                                     subjects_dir=subjects_dir, smooth=smooth)
    
    morph.save(path, overwrite=True)
    
    return morph


#%% Splithalf reliability: within vs between subject correlations

# Get fsaverage src vertices
src_fsaverage = get_fsaverage_src(SUBJECTS_DIR)
fsaverage_vertices = [src_fsaverage[0]['vertno'], src_fsaverage[1]['vertno']]

# Saving arrays
within_corr = []
between_corr = []

# Cache ROI F-value vectors per subject to avoid reloading in inner loop
subject_morphed_cache = {}

for subject in SUBJECT_LIST:

    print(f'Processing subject {subject}...')

    fpath = os.path.join(CONTRAST_MAP_DIR, subject)
    fname_cluster = 'cluster_stats_gradonly.npy' if use_gradonly else 'cluster_stats_base.npy'
    cluster_stats_splithalf = np.load(os.path.join(fpath, fname_cluster), allow_pickle=True)[()]

    # Load F-maps for first and last half
    f_first, _, _, _ = cluster_stats_splithalf['first']
    f_last, _, _, _ = cluster_stats_splithalf['last']

    # Collapse over language time window → (n_vertices,)
    f_first = np.mean(f_first[language_samples[0]:language_samples[1], :], axis=0)
    f_last  = np.mean(f_last[language_samples[0]:language_samples[1], :], axis=0)

    # Morph the last fmap to fsaverage space
    src_subject = mne.read_source_spaces(os.path.join(SOURCE_SPACE_DIR, subject, 
                                                      f'src_{subject}-oct6-src.fif'))
    
    vertices_subject = [src_subject[0]['vertno'], src_subject[1]['vertno']]
    morph_to = get_morph_to_fsaverage(subject, src_subject, src_fsaverage, 
                                      SUBJECTS_DIR, MORPH_DIR)
    
    stc_fmap_last = mne.SourceEstimate(data=f_last,
                                  vertices=vertices_subject,
                                  tmin=tmin, tstep=tstep,
                                  subject=subject)
    
    stc_fsavg_last = morph_to.apply(stc_fmap_last)
    
    # Cache the 'last' half ROI vector for use in between-subject loop
    subject_morphed_cache[subject] = stc_fsavg_last

    # Within-subject correlation across ROIs (first half vs last half)
    res = scipy.stats.spearmanr(f_first, f_last)
    within_corr.append(res.statistic)
    
    
# Between-subject correlations: subject's first half vs other subject's last half
for subject in SUBJECT_LIST:

    fpath = os.path.join(CONTRAST_MAP_DIR, subject)
    fname_cluster = 'cluster_stats_gradonly.npy' if use_gradonly else 'cluster_stats_base.npy'
    cluster_stats_splithalf = np.load(os.path.join(fpath, fname_cluster), allow_pickle=True)[()]
    
    f_first, _, _, _ = cluster_stats_splithalf['first']
    f_first = np.mean(f_first[language_samples[0]:language_samples[1], :], axis=0)
    
    # Morph the first fmap to fsaverage space
    src_subject = mne.read_source_spaces(os.path.join(SOURCE_SPACE_DIR, subject, 
                                                      f'src_{subject}-oct6-src.fif'))
    
    vertices_subject = [src_subject[0]['vertno'], src_subject[1]['vertno']]
    morph_to = get_morph_to_fsaverage(subject, src_subject, src_fsaverage, 
                                      SUBJECTS_DIR, MORPH_DIR)
    
    stc_fmap_first = mne.SourceEstimate(data=f_first,
                                  vertices=vertices_subject,
                                  tmin=tmin, tstep=tstep,
                                  subject=subject)
    
    stc_fsavg_first = morph_to.apply(stc_fmap_first)

    for other_subject in SUBJECT_LIST:
        if subject != other_subject:
            stc_fsavg_last = subject_morphed_cache[other_subject]
            res = scipy.stats.spearmanr(stc_fsavg_first.data, stc_fsavg_last.data)
            between_corr.append(res.statistic)


#%% Summary of results

mu_within  = np.mean(within_corr)
mu_between = np.mean(between_corr)

print(f'Within corr avg rho:  {mu_within:.3f}')
print(f'Within corr std rho:  {np.std(within_corr):.3f}')
print(f'Between corr avg rho: {mu_between:.3f}')
print(f'Between corr std rho: {np.std(between_corr):.3f}')

# Plot distributions
color_within  = 'steelblue'
color_between = 'lightsteelblue'
bins = np.arange(-0.75, 1, 0.05)

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

hist_between, bins_between, p_between = ax.hist(between_corr, bins=bins, density=True, color=color_between, label='Between')
hist_within, bins_within, p_within = ax.hist(within_corr, bins=bins, density=True, color=color_within, label='Within')

for item_between, item_within in zip(p_between, p_within):
    item_between.set_height(item_between.get_height() / max(hist_between))
    item_within.set_height(item_within.get_height()   / max(hist_within))

ax.set_xlabel('Spearman correlation')
ax.set_ylim([0, 1.19])
ax.axvline(x=mu_within, ymax=1.05/1.19, color='darkslategray', linestyle='--', linewidth=1.2)
ax.axvline(x=mu_between, ymax=1.05/1.19, color='slategray', linestyle='--', linewidth=1.2)
ax.legend()
ax.set_ylabel('Relative frequency')

stat, pval = scipy.stats.ranksums(np.ravel(within_corr), between_corr)
print(f'Test statistic: {stat}')
print(f'P-value:        {pval}')

corr_low, corr_high = mu_between, mu_within
min_val, max_val = 1.05, 1.10
ax.plot([corr_low, corr_low, corr_high, corr_high], [min_val, max_val, max_val, min_val], lw=1.5, c='k')
ax.text((corr_low + corr_high) * 0.5, max_val, convert_pvalue_to_asterisks(pval), ha='center', va='bottom', c='k')
sns.despine(trim=True)
