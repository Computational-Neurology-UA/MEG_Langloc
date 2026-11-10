import numpy as np
import mne
import os
import gc

# Sensor data path
subject = 's1'
sample_dir = './data/preproc/dutch/sensor_data'
epoch_fname = f'funcloc_preproc_{subject}-epo.fif'
epoch_fpath = os.path.join(sample_dir, epoch_fname)
epochs = mne.read_epochs(epoch_fpath)

# Freesurfer files path
subjects_dir = '/mnt/d/MEG_LangLoc/Dutch/MRI/Freesurfer'

# Save source data path
save_dir_stc = f'/mnt/d/MEG_LangLoc/Dutch/source_estimate_base/{subject}'
save_dir_stc_erp = f'/mnt/d/MEG_LangLoc/Dutch/source_estimate_erp/{subject}'
save_dir_src = f'/mnt/d/MEG_LangLoc/Dutch/source_space/{subject}'

# Epoch parameter settings
base = (-0.05, 0.05)
amount_words = 12


#%% Setup source space

# Surface-based source space
src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir)

# Save src space
src_fname = f'src_{subject}-oct6-src.fif'
src_fpath = os.path.join(save_dir_src, src_fname)
# src.save(src_fpath, overwrite=True)


#%% Visually confirm source space setup

plot_bem_kwargs = dict(subject=subject, subjects_dir=subjects_dir,
                       brain_surfaces="white", orientation="coronal",
                       slices=[25, 50, 75, 100, 125, 150, 175, 200])

mne.viz.plot_bem(src=src, **plot_bem_kwargs)

fig = mne.viz.plot_alignment(subject=subject, subjects_dir=subjects_dir,
                             surfaces="white", coord_frame="mri", src=src)
fig.plotter.set_background('white')

mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                    distance=0.30, focalpoint=(-0.03, -0.01, 0.03))


#%% Create conductor model

# Single-layer BEM appropriate for MEG
conductivity = [0.3]

# Create BEM surfaces
model = mne.make_bem_model(subject=subject, conductivity=conductivity, 
                           subjects_dir=subjects_dir)

# Turn BEM surfaces into conductor model
bem = mne.make_bem_solution(model)


#%% Compute forward solution

# Load in transformation file
trans_fpath = f'./data/preproc/dutch/trans/trans_{subject}-trans.fif'

# Create forward solution
fwd = mne.make_forward_solution(epoch_fpath,
                                trans=trans_fpath,
                                src=src,
                                bem=bem,
                                meg=True,
                                eeg=False,
                                n_jobs=None,
                                verbose=True)

#%% Create inverse operator

# Read in noise covariance matrix
noise_cov = mne.read_cov(f'./data/preproc/dutch/noise_cov/noise_cov_{subject}-cov.fif')

# Rank from info and empirical rank
rank_info = mne.compute_rank(epochs, rank='info')
rank_empirical = mne.compute_rank(epochs, tol_kind="relative", tol=1e-3)

rank_min = {'meg': min(rank_info['meg'], rank_empirical['meg'])}

# If inverse operator throws a rank-related warning
# rank_min = {'meg': 67}

# Create inverse operator
inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                          fwd,
                                                          noise_cov,
                                                          loose=0.2,
                                                          depth=0.8, 
                                                          rank=rank_min)

#%% Parse sentence-level epochs into word-level epochs

# Equalize event counts for epochs
epochs.equalize_event_counts()

# Define saving structure
epochs_word = {'first': [], 'last': []}

# Partition data in two halves
for condition in ['s1', 'nw1']:
    
    epochs_first = epochs[condition][:len(epochs[condition])//2].get_data()
    epochs_last = epochs[condition][len(epochs[condition])//2:].get_data()

    # Slice sentence epochs into word segments
    for half, epochs_split in zip(['first', 'last'], [epochs_first, epochs_last]):
        
        # Create empty arrays for saving
        amount_trials = len(epochs_split)
        epoch_time_window = (-0.4, 0.4)
        epoch_samples = int(np.rint(500*(epoch_time_window[1] - epoch_time_window[0]))) + 1
        epochs_first_start = int(np.rint(500*(0.5+epoch_time_window[0])))
        
        epochs_split_word_data = np.zeros((amount_trials*amount_words, len(epochs_split[0]), epoch_samples))
        
        # Separate epochs in word contributions
        for index in range(amount_words):
            epochs_split_word_data[amount_trials*index:amount_trials*(index+1), :, :] = epochs_split[:, :, epochs_first_start+200*index:epochs_first_start+epoch_samples+200*index]
        
        # Manually create new epochs instance from sliced data
        epochs_split_word = mne.EpochsArray(data=epochs_split_word_data,
                                            info=epochs.info,
                                            event_id={condition:1})
        # Time shift and crop
        epochs_split_word = epochs_split_word.shift_time(epoch_time_window[0])
        epochs_split_word = epochs_split_word.apply_baseline(base)
        epochs_split_word.crop(tmin=base[0], tmax=None)
        
        # Save condition splithalf partition 
        epochs_word[half].append(epochs_split_word)
        

#%% Compute inverse solution on word-level epochs

# Define parameter settings for single-trial epochs (snr=1)
method = 'dSPM'
snr = 1.0
lambda2 = 1.0/snr**2

# Obtain source time courses
for split in ['first', 'last']:
    for condition, condition_name in zip([0, 1], ['sent', 'nw']):
        
        stc = mne.minimum_norm.apply_inverse_epochs(epochs_word[split][condition],
                                                    inverse_operator,
                                                    lambda2,
                                                    method=method,
                                                    pick_ori=None,
                                                    verbose=True)
        
        for idx, source_estimate in enumerate(stc):
            stc_fname = f'stc_{split}_{condition_name}_{idx}'
            stc_fpath = os.path.join(save_dir_stc, stc_fname)
            source_estimate.save(stc_fpath, ftype='stc', overwrite=True)
        
        del stc
        gc.collect()


#%% Sanity check: Compute the evoked response and look at source reconstruction

# Saving arrays
evoked_word = {'first': [], 'last': []}

# Source reconstruction settings
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2

# Compute evoked responses
for split in ['first', 'last']:
    for condition, condition_name in zip([0, 1], ['sent', 'nw']):
        
        # Save evoked data
        evoked = epochs_word[split][condition].average()
        evoked.plot_white(noise_cov)
        evoked_word[split].append(evoked) 
        
        # Create inverse operator
        inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info,
                                                                  fwd,
                                                                  noise_cov,
                                                                  loose=0.2,
                                                                  depth=0.8, 
                                                                  rank=rank_min)
        # Create STC
        stc_evoked, residual = mne.minimum_norm.apply_inverse(evoked,
                                                              inverse_operator,
                                                              lambda2,
                                                              method=method,
                                                              pick_ori=None,
                                                              return_residual=True,
                                                              verbose=True)
        # Save STC
        stc_fname = f'stc_evoked_{split}_{condition_name}'
        stc_fpath = os.path.join(save_dir_stc_erp, stc_fname)
        stc_evoked.save(stc_fpath, ftype='stc', overwrite=True)


#%% Create inverse operator and plot for specific conditions

split = 'last'
condition = 0

# Inverse operator
inverse_operator = mne.minimum_norm.make_inverse_operator(evoked_word[split][condition].info,
                                                          fwd,
                                                          noise_cov,
                                                          loose=0.2,
                                                          depth=0.8, 
                                                          rank=rank_min)

method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2

stc_evoked, residual = mne.minimum_norm.apply_inverse(evoked_word[split][condition],
                                                      inverse_operator,
                                                      lambda2,
                                                      method=method,
                                                      pick_ori=None,
                                                      return_residual=True,
                                                      verbose=True)

vertno_max, time_max = stc_evoked.get_peak(hemi="lh")

# surface can be 'white', 'pial', 'inflated', ...
surfer_kwargs = dict(hemi="lh", subjects_dir=subjects_dir, views="lateral",
                     initial_time=time_max, time_unit="s", size=(800, 800),
                     smoothing_steps=10, surface='inflated')

brain = stc_evoked.plot(**surfer_kwargs)

brain.add_foci(vertno_max, coords_as_verts=True, hemi="lh", color="blue",
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, "dSPM (plus location of maximal activation)", 
               "title", font_size=14)

