import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test

# Settings
language = 'dutch'
working_dir = f'./data/preproc/{language}/sensor_data'

picks = 'grad'

base = (-0.05, 0.05)
alpha = 0.05
n_permutations = 1000
seed = 30

all_files = os.listdir(working_dir)
epoch_files = [os.path.join(working_dir, f) for f in all_files if f.endswith('-epo.fif')]

# Save results
all_evokeds = {'Sentence': [], 'Nonword': []}


#%% Loop over all epochs and create evokeds

for epoch_file in epoch_files:
    
    # Read in epoch and fetch data
    epochs = mne.read_epochs(epoch_file)
    
    for condition, condition_name in zip(['s1', 'nw1'], ['Sentence', 'Nonword']):
        
        epochs_prestim = epochs[condition]
        epochs_prestim_data = epochs_prestim.get_data(picks=picks)
        
        # Initialize empty arrays
        amount_sentences = len(epochs_prestim_data)
        amount_words = 8 if len(epochs_prestim_data) > 80 else 12
        
        epochs_prestim_data_word = np.zeros((amount_sentences*amount_words, len(epochs_prestim_data[0]), 401))
        
        # Separate epochs in word contributions recursively
        for index in range(amount_words):
            epochs_prestim_data_word[amount_sentences*index:amount_sentences*(index+1), :, :] = epochs_prestim_data[:, :, 50+200*index:451+200*index]
    
        # Manually create a new word-level epochs instance
        epochs_prestim_word = mne.EpochsArray(data=epochs_prestim_data_word, 
                                              info=mne.pick_info(epochs_prestim.info, sel=mne.pick_types(epochs_prestim.info,meg=picks)))
        
        # Shift time axis and baseline epochs
        epochs_prestim_word = epochs_prestim_word.shift_time(-0.4)
        epochs_prestim_word = epochs_prestim_word.apply_baseline(base)
        
        # Create evoked responses
        evoked_prestim_word = epochs_prestim_word.copy().average().crop(tmin=base[0], tmax=None)
        
        # Store evoked instance in predefined array
        all_evokeds[condition_name].append(evoked_prestim_word)


#%% Group-level cluster permutation test (time-only, paired)

sent_evokeds = all_evokeds['Sentence']
nw_evokeds   = all_evokeds['Nonword']
times = sent_evokeds[0].times

def gfp(evoked):
    x = evoked.data
    x = x - x.mean(axis=0, keepdims=True)
    return np.sqrt((x ** 2).mean(axis=0))

X_sent = np.vstack([gfp(e) for e in sent_evokeds])
X_nw   = np.vstack([gfp(e) for e in nw_evokeds])
X_diff = X_sent - X_nw

T_obs, clusters, cluster_pvals, H0 = permutation_cluster_1samp_test(
    X_diff,
    n_permutations=n_permutations,
    tail=0,
    threshold=None,
    seed=seed)

# Collect significant clusters
sig_clusters = [clusters[i] for i, p in enumerate(cluster_pvals) if p < alpha]

print(f"Participants: {X_diff.shape[0]}")
print(f"Significant clusters (alpha={alpha}): {len(sig_clusters)}")

# Plot GFP evokeds and shade significant clusters
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
mne.viz.plot_compare_evokeds(
    all_evokeds,
    combine='gfp',
    axes=ax)

ax.set_ylim(0, 13)
ax.set_yticks(np.arange(0, 13, 2))
ax.spines['left'].set_bounds(0, 12)

# Shade significant clusters
for cl in sig_clusters:
    ax.axvspan(
        times[cl[0][0]],
        times[cl[0][-1]],
        alpha=0.2,
        color='grey'
    )

ax.tick_params(labelsize=14)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.title.set_size(16)

plt.show()