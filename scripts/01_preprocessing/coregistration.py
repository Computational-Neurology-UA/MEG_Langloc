import mne


# Paths for Freesurfer reconstruction
subjects_dir = '/mnt/d/MEG_LangLoc/Dutch/MRI/Freesurfer'
subject = 's1'

# Note: first create BEM before coregistering

#%% Create coregistration

# Trans file will be saved through GUI 
mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)


#%% Visual check of coregistration transformation

# The transformation file obtained by coregistration
trans_fpath = f'./data/preproc/dutch/trans/trans_{subject}-trans.fif'

info = mne.io.read_info(f'./data/preproc/dutch/sensor_data/funcloc_preproc_{subject}-epo.fif')

mne.viz.plot_alignment(info, trans_fpath, subject=subject, dig=True,
                       meg=["helmet", "sensors"], subjects_dir=subjects_dir, 
                       surfaces="head-dense")


