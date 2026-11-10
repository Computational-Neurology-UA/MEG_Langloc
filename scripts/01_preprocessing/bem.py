import mne


# Paths to Freesurfer files
subjects_dir = '/mnt/d/MEG_LangLoc/Dutch/MRI/Freesurfer'

subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20]

#%% Create BEM 

for subject_id in subjects:
    
    subject = f's{subject_id}'
    
    # BEM will automatically be saved in Freesurfer dir
    mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir, show=True, overwrite=True)
    mne.bem.make_scalp_surfaces(subject, subjects_dir=subjects_dir, force=True, overwrite=True)
