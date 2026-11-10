# Functional identification of language-responsive sensors in MEG

Analysis code accompanying:

**Functional identification of language-responsive sensors in individual participants in MEG investigations**

Mathias Huybrechts, Rose Bruffaerts, Alvincé Le Arnz Pongos, Cory Shain, Benjamin Lipkin, Matthew Siegelman, Vincent Wens, Martin Sjøgård, Idan Asher Blank, Serge Goldman, Xavier De Tiège, and Evelina Fedorenko.

*Journal of Neurophysiology* **136**, 1075–1085 (2026).  
https://doi.org/10.1152/jn.00628.2025

Mathias Huybrechts and Rose Bruffaerts contributed equally.

## Overview

This repository contains analysis scripts for identifying language-responsive MEG sensors and cortical sources at the level of individual participants.

The study uses a sentence-versus-nonword contrast. Participants read sequences of sentences and pronounceable nonwords while MEG activity was recorded. The contrast is used to identify responses associated with higher-level language processing.

A central question is whether language-responsive sensors and sources are more consistent within an individual than across individuals. The analyses therefore compare participant-specific functional regions of interest (sensors or sources of interest; SOIs) with SOIs defined from group-averaged data.

## Repository organisation

The scripts are grouped by their role in the analysis. The suggested order below describes the broad workflow; the exact execution order may depend on the input files and intermediate outputs expected by each script.

```text
.
├── README.md
└── scripts/
    ├── 01_preprocessing/
    │   ├── bem.py
    │   ├── coregistration.py
    │   ├── erf_grandavg.py
    │   └── noise_covariance.py
    ├── 02_sensor_space/
    │   ├── sensor_spatiotemporal_clusterest.py
    │   └── sensor_splithalf_analysis.py
    ├── 03_source_space/
        ├── source_reconstruction.py
        ├── source_splithalf_withinbetween.py
        ├── source_splithalf_effectsize.py
        ├── source_splithalf_groupF_viz.py
        └── source_splithalf_individualF_viz.py
```

### Preprocessing and anatomical preparation

| Script | Purpose |
|---|---|
| `bem.py` | Creates the boundary element model (BEM) used for source modelling. |
| `coregistration.py` | Performs coregistration between the participant's MRI and MEG coordinate systems. |
| `erf_grandavg.py` | Computes grand-average event-related fields (ERFs), including the responses shown in Figure 1. |
| `noise_covariance.py` | Computes the noise covariance matrix used in source estimation. |

### Sensor-space analyses

| Script | Purpose |
|---|---|
| `sensor_spatiotemporal_clusterest.py` | Runs the sensor-space spatiotemporal cluster test reported in the supplementary materials. |
| `sensor_splithalf_analysis.py` | Runs the sensor-space split-half analyses, including the analyses contributing to Figures 2–4. |

### Source-space analyses

| Script | Purpose |
|---|---|
| `source_reconstruction.py` | Computes source estimates from the MEG data. |
| `source_splithalf_withinbetween.py` | Performs the within-participant and between-participant split-half analyses in source space; contributes to Figure 2. |
| `source_splithalf_effectsize.py` | Computes source-space effect sizes; contributes to Figure 4. |
| `source_splithalf_groupF_viz.py` | Generates group-space brain visualisations and group-averaged SOIs; contributes to Figure 3. |
| `source_splithalf_individualF_viz.py` | Generates brain visualisations and SOIs in participants' native spaces; contributes to Figure 3. |

## Study and analysis summary

The study uses two datasets:

- **Dutch-speaking cohort:** 19 participants, with sensor- and source-space analyses.
- **English-speaking cohort:** 23 participants were recruited; 20 were retained for analysis. The reported analyses were conducted in sensor space.

Participants read 12-word sentences and sequences of pronounceable nonwords, presented one item at a time using rapid serial visual presentation. The presentation rate was 385 ms per item in the Dutch experiment and 400 ms per item in the English experiment.

The main contrast was **sentences vs nonwords**. At each sensor or source location and time point, condition differences were quantified using an F statistic from a one-way ANOVA. The main analyses focused on the 100–350 ms interval following word or nonword onset.

For the split-half analyses, trials were divided into first and second halves. The analyses compared the spatial consistency of the resulting maps within participants and between participants. SOIs were also defined using one half of the data and evaluated on the held-out half.

The paper reports greater spatial consistency within participants than between participants, substantial variation in language-responsive topographies across participants, and higher held-out responses for individually defined SOIs than for group-defined SOIs.

## Methods and software

The processing described in the paper included temporal signal space separation (tSSS), head-motion correction, filtering, downsampling, epoching, artifact rejection, and ICA-based artifact removal. Source-space analyses used individual structural MRI data, FreeSurfer surfaces, a single-shell BEM, and dynamic statistical parametric mapping (dSPM).

The software reported for the study includes:

- [MNE-Python](https://mne.tools/) 1.11.0
- MaxFilter 2.2
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [Wuggy](https://github.com/crr-ugent/wuggy) for Dutch pseudoword generation

MaxFilter and FreeSurfer are external software dependencies, not Python packages installed through `pip`.

## Data and materials

The task stimuli associated with the study are available through the OSF project. The [supplementary materials](https://doi.org/10.17605/OSF.IO/VC2BW) provide additional analyses.

Raw MEG data are available from the authors upon reasonable request.

## Citation

If you use these scripts or build on the analyses, please cite the paper:

```bibtex
@article{huybrechts2026functional,
  title   = {Functional identification of language-responsive sensors in individual participants in MEG investigations},
  author  = {Huybrechts, Mathias and Bruffaerts, Rose and
             Pongos, Alvinc{\'e} Le Arnz and Shain, Cory and
             Lipkin, Benjamin and Siegelman, Matthew and Wens, Vincent and
             Sj{\o}g{\aa}rd, Martin and Blank, Idan Asher and
             Goldman, Serge and De Ti{\`e}ge, Xavier and Fedorenko, Evelina},
  journal = {Journal of Neurophysiology},
  volume  = {136},
  pages   = {1075--1085},
  year    = {2026},
  doi     = {10.1152/jn.00628.2025}
}
```

## Contact

For questions about the study or its materials, contact:

- Rose Bruffaerts - rose.bruffaerts [at] uantwerpen [dot] be
- Evelina Fedorenko - evelina9 [at] mit [dot] edu
