import matplotlib.pyplot as plt
import numpy as np
import mne
import os

# Load directory
SUBJECTS_DIR = '/mnt/d/MEG_LangLoc/Dutch/MRI/Freesurfer'
LOAD_DIR = './data/preproc/dutch/source_SOI_results'
fpath_first = os.path.join(LOAD_DIR, 'group_fmap_fsaverage_first_base.npy')
fpath_last = os.path.join(LOAD_DIR, 'group_fmap_fsaverage_last_base.npy')
OUTPUT_DIR = './data/preproc/dutch/source_SOI_results'

# Parameters
TMIN_SOI = 0.100
TMAX_SOI = 0.350
SOI_PERCENTILE = 90

# Timing of the original STCs
TMIN_STC = -0.05
TSTEP_STC = 0.002

# Plotting params
SURF = 'inflated'
BACKGROUND = 'white'
HEMI = 'split'
COLORMAP = 'hot'
SMOOTH_VIZ = 5
SAVE_OUTPUT = True


#%% Define helper function

def time_average_fmap(F_obs, tmin, tstep, twin_min, twin_max):
    """
    F_obs : (n_times, n_sources) — spatio_temporal_cluster_test convention
    Returns (n_sources,) — mean F across the time window
    """
    times = tmin + np.arange(F_obs.shape[0]) * tstep
    tmask = (times >= twin_min) & (times <= twin_max)
    return F_obs[tmask, :].mean(axis=0)


def split_lh_rh(data_flat, src):
    """Split flat (n_sources,) array into lh and rh parts."""
    n_lh = len(src[0]['vertno'])
    return data_flat[:n_lh], data_flat[n_lh:]


#%% Helper functions

def plot_brain(subject, subjects_dir, fmap_lh, fmap_rh, vertices_lh, vertices_rh,
               fmin, fmid, fmax, title, make_screenshot=False):
    brain = mne.viz.Brain(subject=subject, hemi=HEMI, surf=SURF,
                          subjects_dir=subjects_dir, background=BACKGROUND,
                          title=title, size=(1200,600))
    for arr, verts, h in [(fmap_lh, vertices_lh, 'lh'),
                           (fmap_rh, vertices_rh, 'rh')]:
        brain.add_data(array=arr, vertices=verts, hemi=h,
                       colormap=COLORMAP, fmin=fmin, fmid=fmid, fmax=fmax,
                       transparent=True, smoothing_steps=SMOOTH_VIZ,
                       colorbar=True)
        
    if make_screenshot:
        create_screenshot(brain, title)
        
    return brain

def create_screenshot(brain, title):

    # Extract screenshot with alpha channel (RGBA)
    screenshot = brain.screenshot(mode='rgba')

    # Crop using alpha channel
    alpha = screenshot[:, :, 3]
    nonempty = alpha > 0
    nonempty_row = nonempty.any(1)
    nonempty_col = nonempty.any(0)
    cropped_screenshot = screenshot[nonempty_row][:, nonempty_col]

    # Create transparent figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.patch.set_alpha(0)
    
    ax.set_facecolor((0, 0, 0, 0))
    ax.imshow(cropped_screenshot)
    ax.axis('off')

    # Output filename
    fname = title.replace(' - ', '_').replace(' ', '_') + '_rev.png'
    plt.savefig(os.path.join(OUTPUT_DIR, 'subjects', 'group_avg', fname),
                dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)

    plt.close(fig)

    return None

def add_soi_outline(brain, mask, src, subject,
                    color='black',
                    smooth=1):

    """
    Add a smooth parcel outline from a binary source mask.
    """

    # Convert binary mask to STC
    stc = mne.SourceEstimate(
        data=mask.astype(float)[:, np.newaxis],
        vertices=[src[0]['vertno'], src[1]['vertno']],
        tmin=-0.05,
        tstep=0.002,
        subject=subject)
    

    # Convert STC -> labels
    labels_lh, labels_rh = mne.stc_to_label(
        stc,
        src=src,
        smooth=smooth,
        connected=True,
        subjects_dir=SUBJECTS_DIR)

    # Add left hemi labels
    for label in labels_lh:
        brain.add_label(
            label,
            borders=2,
            color=color, 
            alpha=0.95)

    # Add right hemi labels
    for label in labels_rh:
        brain.add_label(
            label,
            borders=2,
            color=color,
            alpha=0.95)

#%% Load in data

# Load in group fmap
group_fmap_first = np.load(fpath_first, allow_pickle=True)
group_fmap_last = np.load(fpath_last, allow_pickle=True)

# Setup source space
src = mne.setup_source_space("fsaverage", spacing="oct6", subjects_dir=None, 
                             add_dist=False, verbose=False)

# Extract vertices
vertices_lh = src[0]["vertno"]
vertices_rh = src[1]["vertno"]

# Time-average group Fmap
group_fmap_avg_first = time_average_fmap(group_fmap_first.T, TMIN_STC, TSTEP_STC, TMIN_SOI, TMAX_SOI)
group_fmap_avg_last = time_average_fmap(group_fmap_last.T, TMIN_STC, TSTEP_STC, TMIN_SOI, TMAX_SOI)


#%% Inspect entire fmap per half

for half, fmap in zip(['first', 'last'], [group_fmap_avg_first, group_fmap_avg_last]):
    
    # Split in hemisphere vertices
    fmap_lh_p1, fmap_rh_p1 = split_lh_rh(fmap, src)
    
    # Get plotting limits
    fmin_p1 = float(np.percentile(fmap, 50))
    fmid_p1 = float(np.percentile(fmap, 75))
    fmax_p1 = float(fmap.max())

    print(f"  Colour scale: fmin={fmin_p1:.3f}, fmid={fmid_p1:.3f}, fmax={fmax_p1:.3f}")
    
    brain1 = plot_brain('fsaverage', 
                        fmap_lh_p1, fmap_rh_p1,
                        vertices_lh, vertices_rh, 
                        fmin_p1, fmid_p1, fmax_p1,
                        title=f'Group - Full Fmap - {half} half',
                        make_screenshot=SAVE_OUTPUT)
    
#%% Inspect entire fmap per half with SOI outline

for half, fmap in zip(['first', 'last'], [group_fmap_avg_first, group_fmap_avg_last]):
    
    # Threshold top percentile
    threshold_indiv = np.percentile(fmap, SOI_PERCENTILE)
    F_avg_indiv = fmap.copy()
    mask = F_avg_indiv > threshold_indiv
    
    # Split in hemisphere vertices
    fmap_lh_p1, fmap_rh_p1 = split_lh_rh(fmap, src)
    
    # Get plotting limits
    fmin_p1 = float(np.percentile(fmap, 50))
    fmid_p1 = float(np.percentile(fmap, 75))
    fmax_p1 = float(fmap.max())
   
    print(f"  Colour scale: fmin={fmin_p1:.3f}, fmid={fmid_p1:.3f}, fmax={fmax_p1:.3f}")
    
    brain1 = plot_brain('fsaverage', 
                        SUBJECTS_DIR, 
                        fmap_lh_p1, fmap_rh_p1,
                        vertices_lh, vertices_rh, 
                        fmin_p1, fmid_p1, fmax_p1,
                        title=f'group - Full Fmap - {half} half',
                        make_screenshot=SAVE_OUTPUT)
    
    # Add SOI outlines
    add_soi_outline(brain1, mask, src, subject='fsaverage', color='black',
                    smooth=True)
    # Optional screenshot
    if SAVE_OUTPUT:
        create_screenshot(brain1, f'group_combined_SOI_{half}')

#%% Insepct binarized SOIs per half

f_binarized = {}

for half, fmap in zip(['first', 'last'], [group_fmap_avg_first, group_fmap_avg_last]):
    
    # Threshold top percentile
    threshold_indiv = np.percentile(fmap, SOI_PERCENTILE)
    F_avg_indiv = fmap.copy()
    F_avg_indiv[F_avg_indiv < threshold_indiv] = np.nan
    
    print(f"  Threshold={threshold_indiv:.3f}  |  "
          f"{np.isfinite(F_avg_indiv).sum()} sources selected")
    
    # Binarize individual soi
    F_binarized_indiv = fmap.copy()
    F_binarized_indiv = F_binarized_indiv > threshold_indiv
    f_binarized[half] = F_binarized_indiv
    
    # Split in hemisphere vertices
    fmap_lh_p2, fmap_rh_p2 = split_lh_rh(F_avg_indiv, src)
    
    brain2 = plot_brain('fsaverage', 
                        fmap_lh_p2, fmap_rh_p2,
                        vertices_lh, vertices_rh,
                        fmin=float(threshold_indiv),
                        fmid=float(np.nanmedian(F_avg_indiv)),
                        fmax=float(np.nanmax(F_avg_indiv)),
                        title=f'Group - Individual SOI p{SOI_PERCENTILE} - {half} half',
                        make_screenshot=SAVE_OUTPUT)
