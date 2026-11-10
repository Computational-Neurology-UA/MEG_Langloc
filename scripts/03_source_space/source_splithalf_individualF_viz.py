import matplotlib.pyplot as plt
import os
import numpy as np
import mne

# Subject settings
subject = 's13'

# Paths 
SUBJECTS_DIR = '/mnt/d/MEG_LangLoc/Dutch/MRI/Freesurfer'
SOURCE_SPACE_DIR = '/mnt/d/MEG_LangLoc/Dutch/source_space'
CONTRAST_MAP_DIR = './data/preproc/dutch/contrast_map'
MORPH_DIR = './data/preproc/dutch/morphs'
OUTPUT_DIR = './data/preproc/dutch/source_SOI_results'

# Parameters
TMIN_SOI = 0.100
TMAX_SOI = 0.350
SOI_PERCENTILE = 90
MORPH_SMOOTH = 5

# Timing of the original STCs
TMIN_STC = -0.05
TSTEP_STC = 0.002

# Plotting params
SURF = 'inflated'
BACKGROUND = 'white'
HEMI = 'split'
COLORMAP = 'hot'
SMOOTH_VIZ = 5
SAVE_OUTPUT= True


#%% Helper functions

def time_average_fmap(F_obs, tmin, tstep, twin_min, twin_max):
    """Average F_obs (n_times, n_sources) over a time window -> (n_sources,)."""
    times = tmin + np.arange(F_obs.shape[0]) * tstep
    tmask = (times >= twin_min) & (times <= twin_max)
    return F_obs[tmask, :].mean(axis=0)


def split_lh_rh(data_flat, src):
    """Split flat (n_sources,) array into lh and rh parts."""
    n_lh = len(src[0]['vertno'])
    return data_flat[:n_lh], data_flat[n_lh:]


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
    
    # brain.add_text(0.2, 0.9, title)
    
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'subjects', subject, fname),
                dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)

    plt.close(fig)

    return None

def dice_coefficient(a, b):

    intersection = np.sum(a & b)
    size_a = np.sum(a)
    size_b = np.sum(b)
    
    return 100 * 2 * intersection / (size_a + size_b)

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

# Subject source space
src_subject  = mne.read_source_spaces(os.path.join(SOURCE_SPACE_DIR, subject, 
                                                   f'src_{subject}-oct6-src.fif'))
vertices_lh  = src_subject[0]['vertno']
vertices_rh  = src_subject[1]['vertno']

# Native-space F-map for this subject / half
cluster_stats = np.load(os.path.join(CONTRAST_MAP_DIR, subject, 'cluster_stats_base.npy'), allow_pickle=True).item()
F_obs_first = cluster_stats['first'][0]
F_obs_last = cluster_stats['last'][0]

F_avg_first = time_average_fmap(F_obs_first, TMIN_STC, TSTEP_STC, TMIN_SOI, TMAX_SOI)
F_avg_last = time_average_fmap(F_obs_last, TMIN_STC, TSTEP_STC, TMIN_SOI, TMAX_SOI)

# Saved group F-map in fsaverage (from effect-size script)
group_fmap_fsavg_first = np.load(os.path.join(OUTPUT_DIR, 'group_fmap_fsaverage_first_base.npy'), allow_pickle=True)
group_fmap_fsavg_last = np.load(os.path.join(OUTPUT_DIR, 'group_fmap_fsaverage_last_base.npy'), allow_pickle=True)

group_fmap_avg_first = time_average_fmap(group_fmap_fsavg_first.T, TMIN_STC, TSTEP_STC, TMIN_SOI, TMAX_SOI)
group_fmap_avg_last = time_average_fmap(group_fmap_fsavg_last.T, TMIN_STC, TSTEP_STC, TMIN_SOI, TMAX_SOI)

# fsaverage source space (for backprojection)
fsavg_src_path = os.path.join(SUBJECTS_DIR, 'fsaverage', 'bem', 'fsaverage-oct6-src.fif')
src_fsaverage = mne.read_source_spaces(fsavg_src_path)
fsaverage_vertices = [src_fsaverage[0]['vertno'], src_fsaverage[1]['vertno']]

# Cached morph: fsaverage -> subject native
morph_path = os.path.join(MORPH_DIR, f'fsaverage-to-{subject}-morph.h5')
morph_from = mne.read_source_morph(morph_path)


#%% Complete F-map in native subject space

for half, fmap in zip(['first', 'last'], [F_avg_first, F_avg_last]):
    
    # Split in hemisphere vertices
    fmap_lh_p1, fmap_rh_p1 = split_lh_rh(fmap, src_subject)
    
    # Get plotting limits
    fmin_p1 = float(np.percentile(fmap, 50))
    fmid_p1 = float(np.percentile(fmap, 75))
    fmax_p1 = float(fmap.max())
    
    print(f"  Colour scale: fmin={fmin_p1:.3f}, fmid={fmid_p1:.3f}, fmax={fmax_p1:.3f}")
    
    brain1 = plot_brain(subject, 
                        SUBJECTS_DIR, 
                        fmap_lh_p1, fmap_rh_p1,
                        vertices_lh, vertices_rh, 
                        fmin_p1, fmid_p1, fmax_p1,
                        title=f'{subject} - Full Fmap - {half} half',
                        make_screenshot=SAVE_OUTPUT)
    
#%% Complete F-map in native space with parcel top10%

for half, fmap in zip(['first', 'last'], [F_avg_first, F_avg_last]):
    
    # Threshold top percentile
    threshold_indiv = np.percentile(fmap, SOI_PERCENTILE)
    F_avg_indiv = fmap.copy()
    mask = F_avg_indiv > threshold_indiv
    
    # Split in hemisphere vertices
    fmap_lh_p1, fmap_rh_p1 = split_lh_rh(fmap, src_subject)
    
    # Get plotting limits
    fmin_p1 = float(np.percentile(fmap, 50))
    fmid_p1 = float(np.percentile(fmap, 75))
    fmax_p1 = float(fmap.max())

    print(f"  Colour scale: fmin={fmin_p1:.3f}, fmid={fmid_p1:.3f}, fmax={fmax_p1:.3f}")
    
    brain1 = plot_brain(subject, 
                        SUBJECTS_DIR, 
                        fmap_lh_p1, fmap_rh_p1,
                        vertices_lh, vertices_rh, 
                        fmin_p1, fmid_p1, fmax_p1,
                        title=f'{subject} - Full Fmap - {half} half',
                        make_screenshot=SAVE_OUTPUT)
    
    # Add SOI outlines
    add_soi_outline(brain1, mask, src_subject, subject=subject, color='black',
                    smooth=True)
    # Optional screenshot
    if SAVE_OUTPUT:
        create_screenshot(brain1, f'{subject}_combined_SOI_{half}')

#%% Individual SOI in native subject space

f_binarized = {}

for half, fmap in zip(['first', 'last'], [F_avg_first, F_avg_last]):
    
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
    fmap_lh_p2, fmap_rh_p2 = split_lh_rh(F_avg_indiv, src_subject)
    
    brain2 = plot_brain(subject, 
                        SUBJECTS_DIR, 
                        fmap_lh_p2, fmap_rh_p2,
                        vertices_lh, vertices_rh,
                        fmin=float(threshold_indiv),
                        fmid=float(np.nanmedian(F_avg_indiv)),
                        fmax=float(np.nanmax(F_avg_indiv)),
                        title=f'{subject} - Individual SOI p{SOI_PERCENTILE} - {half} half',
                        make_screenshot=SAVE_OUTPUT)

dice = dice_coefficient(f_binarized['first'], f_binarized['last'])
print(f"  Dice coefficient between two halves={dice:.2f}%")

#%% Group-derived SOI backprojected into native subject space

for half, fmap, fmap_group in zip(['first', 'last'], 
                                  [F_avg_first, F_avg_last],
                                  [group_fmap_avg_first, group_fmap_avg_last]):
 
    # Threshold group fmap in fsaverage -> binary mask
    threshold_group = np.percentile(fmap_group, SOI_PERCENTILE)
    group_soi_mask = fmap_group >= threshold_group
    n_soi_fsavg = int(group_soi_mask.sum())
    
    print(f"  Group SOI threshold={threshold_group:.3f}  |  "
          f"{n_soi_fsavg} sources in fsaverage")
    
    # Morph binary mask to native space
    stc_mask_fsavg = mne.SourceEstimate(data=group_soi_mask.astype(float)[:, np.newaxis],
                                        vertices=fsaverage_vertices, 
                                        tmin=0.0, tstep=1.0, subject='fsaverage')
    
    native_vals = morph_from.apply(stc_mask_fsavg).data[:, 0]
    native_group_mask = native_vals > 0
    
    # Top-N fallback (mirrors effect-size script)
    if native_group_mask.sum() < 0.5 * n_soi_fsavg:
        print(f"  Warning: only {native_group_mask.sum()} vertices after thresholding "
              f"(expected ~{n_soi_fsavg}). Using top-N fallback.")
        native_group_mask = native_vals >= np.sort(native_vals)[-n_soi_fsavg]
    
    print(f"  Group SOI in native space: {int(native_group_mask.sum())} sources")
    
    # Colour SOI vertices by their native F-value; mask the rest as transparent
    F_avg_group = fmap.copy()
    F_avg_group[~native_group_mask] = np.nan
    
    # Split per hemisphere
    fmap_lh_p3, fmap_rh_p3 = split_lh_rh(F_avg_group, src_subject)
    
    brain3 = plot_brain(subject, SUBJECTS_DIR, 
                        fmap_lh_p3, fmap_rh_p3,
                        vertices_lh, vertices_rh,
                        fmin=float(np.nanmin(F_avg_group)),
                        fmid=float(np.nanmedian(F_avg_group)),
                        fmax=float(np.nanmax(F_avg_group)),
                        title=f'{subject} - Group SOI backprojected - {half} half',
                        make_screenshot=SAVE_OUTPUT)

