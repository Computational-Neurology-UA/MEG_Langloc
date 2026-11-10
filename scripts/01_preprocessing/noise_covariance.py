import mne
import os


# Fetch epochs
subject_id = 1
working_dir = './data/preproc/dutch/sensor_data'
file_name = f'funcloc_preproc_s{subject_id}-epo.fif'
epoch_fpath = os.path.join(working_dir, file_name)

#%% Estimate noise covariance matrix

# Read in epochs and baseline 
epochs = mne.read_epochs(epoch_fpath)
epochs = epochs.apply_baseline((-0.5, 0))

# Estimate noise covariance 
noise_cov_baseline = mne.compute_covariance(epochs, 
                                            tmax=0, 
                                            method='auto', 
                                            rank='info')

noise_cov_baseline.plot(epochs.info, proj=True)

# Check whitened data
evoked = epochs.average().crop(-0.5, 1)
evoked.plot_white(noise_cov_baseline, time_unit="s")


#%% Estimate noise covariance matrix empirically

# Check manual rank estimation (contribution of small eigenvalues)
rank_noise = mne.compute_rank(epochs, tol_kind="relative", tol=1e-3)

# Estimate noise covariance 
noise_cov_baseline = mne.compute_covariance(epochs, 
                                            tmax=0, 
                                            method='auto', 
                                            rank=rank_noise)

noise_cov_baseline.plot(epochs.info, proj=True)

# Check whitened data
evoked = epochs.average().crop(-0.5, 1)
evoked.plot_white(noise_cov_baseline, time_unit="s")


#%% Selecting min rank option

# Rank from info and empirical rank
rank_info = mne.compute_rank(epochs, rank='info')
rank_noise = mne.compute_rank(epochs, tol_kind="relative", tol=1e-3)

rank_min = {'meg': min(rank_info['meg'], rank_noise['meg'])}

# Estimate noise covariance 
noise_cov_baseline = mne.compute_covariance(epochs, 
                                            tmax=0, 
                                            method='auto', 
                                            rank=rank_min)

noise_cov_baseline.plot(epochs.info, proj=True)

# Check whitened data
evoked = epochs.average().crop(-0.5, 1)
evoked.plot_white(noise_cov_baseline, time_unit="s")


#%% Save results

file_name_save = f'noise_cov_s{subject_id}-cov.fif'
save_dir = './data/preproc/dutch/noise_cov'
save_path = os.path.join(save_dir, file_name_save)

mne.write_cov(save_path, noise_cov_baseline, overwrite=True)


