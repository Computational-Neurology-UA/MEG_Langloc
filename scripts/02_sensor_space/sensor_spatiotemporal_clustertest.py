from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test
from mne.viz import plot_compare_evokeds
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import shutil
import mne
import os


# Set parameters
language = 'english'
base = (-0.05, 0.05)

random_seed = 30
alpha_cluster_forming = 0.01
verbose = False

# Fetch all epochs
working_dir = f'./data/preproc/{language}/sensor_data'
all_files = os.listdir(working_dir)
epoch_files = [os.path.join(working_dir, f) for f in all_files if f.endswith('-epo.fif')]


#%% Loop spatiotemporal clustertest over all participants

# Read in epochs
for epoch_fpath in epoch_files:
        
    # Define subject-level params
    subject_id = int([item[1:] for item in epoch_fpath.split('_') if item.startswith('s')][0])
    amount_words = 8 if language=='english' and subject_id in [1,2,3,4,5,7] else 12
    
    # Define output save path and clear old results
    filepath_save = f'C:/Users/mathi/Projects/MEG_LangLoc/data/spatiotemporal/{language}/s{subject_id}'
    
    if os.path.isdir(filepath_save):
        shutil.rmtree(filepath_save)
    
    os.makedirs(filepath_save, exist_ok=True)

    # Read in epochs
    epochs = mne.read_epochs(epoch_fpath)
    epochs.equalize_event_counts()

    # Run over magnetometers and gradiometers separately
    for picks in ['grad', 'mag']:
        
        # Define saving structure
        epochs_word = {'first': [], 'last': []}
        
        # Partition data in two halves
        for condition in ['s1', 'nw1']:
            
            epochs_first = epochs[condition][:len(epochs[condition])//2].get_data(picks=picks)
            epochs_last = epochs[condition][len(epochs[condition])//2:].get_data(picks=picks)
            
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
                                                    info=mne.pick_info(epochs.info, sel=mne.pick_types(epochs.info,meg=picks)),
                                                    event_id={condition:1},
                                                    verbose=verbose).pick(picks=picks)
                # Time shift and baseline
                epochs_split_word = epochs_split_word.shift_time(epoch_time_window[0])
                epochs_split_word = epochs_split_word.apply_baseline(base, verbose=verbose)
                epochs_split_word.crop(tmin=base[0], tmax=None)
                
                # Save condition splithalf partition 
                epochs_word[half].append(epochs_split_word)
        
        # Transpose data for spatiotemporal cluster test
        epochs_transposed = {'first': None, 'last': None}
        
        for half in ['first', 'last']:
            data = [epoch_condition.get_data(copy=False) for epoch_condition in epochs_word[half]]
            epochs_transposed[half] = [np.transpose(x, (0, 2, 1)) for x in data]
        
        # Define spatiotemporal cluster test parameters
        cluster_stats_splithalf = {'first': None, 'last': None}
        adjacency, ch_names = find_ch_adjacency(epochs_split_word.info, ch_type=picks)
        
        for half in ['first', 'last']:
            
            # Define clustertest parameters
            n_conditions = 2
            n_observations = len(epochs_transposed[half][0])
            dfn = n_conditions - 1
            dfd = n_observations - n_conditions
            
            # We calculate 1 - alpha_cluster_forming to get the critical value
            f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)
            
            # Run the cluster based permutation analysis
            cluster_stats = spatio_temporal_cluster_test(
                epochs_transposed[half], 
                n_permutations=1000,
                threshold=f_thresh, 
                tail=1,
                n_jobs=8, 
                adjacency=adjacency, 
                seed=random_seed)
            
            # Save results
            cluster_stats_splithalf[half] = cluster_stats
        
        # Save cluster_statistics   
        np.save(os.path.join(filepath_save, f'epochs_word_{picks}.npy'), epochs_word)
        np.save(os.path.join(filepath_save, f'cluster_stats_{picks}.npy'), cluster_stats_splithalf)
        
        # Set parameters for visualization
        p_accept = 0.05
        colors = {'Sentence':'steelblue', 'Nonword':'crimson'}
        
        # Plot splithalf clusters
        for half in ['first', 'last']:
            
            F_obs, clusters, p_values, _ = cluster_stats_splithalf[half]
            good_cluster_inds = np.where(p_values < p_accept)[0]
            
            if len(good_cluster_inds) == 0:
                print(f'No clusters found in {half} half')
                
            # Organize evoked responses for plotting
            evokeds = {'Sentence':epochs_word[half][0].average(), 
                       'Nonword': epochs_word[half][1].average()}
            
            # Loop over clusters
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                
                # Unpack cluster information, get unique indices
                time_inds, space_inds = np.squeeze(clusters[clu_idx])
                ch_inds = np.unique(space_inds)
                time_inds = np.unique(time_inds)
            
                # Get topography for F stats
                f_map = F_obs[time_inds, ...].mean(axis=0)
            
                # Get signals at the sensors contributing to the cluster
                sig_times = epochs_word[half][0].times[time_inds]
            
                # Create spatial mask
                mask = np.zeros((f_map.shape[0], 1), dtype=bool)
                mask[ch_inds, :] = True
            
                # Initialize figures
                fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout='tight')
            
                # Plot average test statistic and mark significant sensors
                f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs_word[half][0].info, tmin=0)
                f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap="Reds",
                                      vlim=(np.min, np.max), show=False, 
                                      colorbar=False, mask_params=dict(markersize=10))
                image = ax_topo.images[0]
            
                # Remove the title that would otherwise say "0.000 s"
                ax_topo.set_title("")
            
                # Create additional axes (for ERF and colorbar)
                divider = make_axes_locatable(ax_topo)
            
                # Add axes for colorbar
                ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(image, cax=ax_colorbar)
                ax_topo.set_xlabel("Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]]))
            
                # Add new axis for time courses and plot time courses
                ax_signals = divider.append_axes("right", size="300%", pad=1.2)
                title = f"S{subject_id} {half} {picks} - Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
                
                if len(ch_inds) > 1:
                    title += "s (mean)"
                
                plot_compare_evokeds(evokeds, title=title, picks=ch_inds, 
                                     axes=ax_signals, colors=colors, show=False,
                                     split_legend=True, truncate_yaxis="auto")
            
                # Plot temporal cluster extent
                ymin, ymax = ax_signals.get_ylim()
                ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1], 
                                         color="orange", alpha=0.3)
                
                plt.savefig(os.path.join(filepath_save, f'{picks}-{half}_cluster_n{i_clu + 1}.png'), dpi=300)
                plt.close(fig)
                