# Constructing The Metabolic Connectome from MRSI

This repository contains all the necessary files to construct a within-subject metabolic similarity matrix (MeSiM) based on MRSI scans, as detailed in the following preprint: [Constructing the Human Brain Metabolic Connectome Using MR Spectroscopic Imaging](https://www.biorxiv.org/content/10.1101/2025.03.10.642332v1).

<img src="figures/Figure1.png" alt="Figure 1">

## License
This project is licensed under the terms described in [LICENSE](./LICENSE).

## Datasets
A ```data/BIDS/Dummy-Project``` has been provided for demonstration purposes. For those interested in obtaining access to the full MRSI dataset beyond demonstration purposes, please reach out to the authors of this paper with a well-motivated request detailing your research aims and how the data will be used. 

## Installation

### Prerequisites

- **Python 3.x**: Ensure you have Python 3 installed on your system.
- **Conda or Miniconda**: Optionally, use Conda/Miniconda for environment management and package installation.
- **[CHIMERA](https://github.com/connectomicslab/chimera)**: An open source framework for combining multiple parcellations



### Install toolbox

1. **Clone the Repository**
   ```bash
   git clone git@github.com:MRSI-Psychosis-UP/MRSI-Metabolic-Connectome.git
   cd MRSI-Metabolic-Connectome
2. **Install package**
    ```bash
    bash build_env.sh

3. **Activate env**
    ```bash
    conda activate mrsitooldemo_env

4. **Set up ENV paths**
   ```python
   python set_env_paths.py
 and navigate to your BIDS folder containing MRI data. Choose data/BIDS for demonstration purposes



## Inputs

### Chimera anatomical parcellation files ###
- Chimera parcellation are already provided for the ```S0001-V1``` subject but can be replicated with
   ```sh
   chimera -b data/BIDS/Dummy-Project/ -d data/BIDS/Dummy-Project/derivatives/ \ 
           --freesurferdir data/BIDS/Dummy-Project/derivatives/freesurfer/     \
           -p LFMFIIFIS -g 2
   ```
- Use ```-p LFMFIIFIS -g 2``` supra-region sequence with 2mm GM grow inside WM in order to replicate the results in the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.10.642332v1).
- Chimera outputs are saved  ```PROJECT_NAME/derivatives/chimera-atlases```

### MRSI files ###
- Reconstrcuted MRSI files should be placed in the following derivatives folder
   ```sh
   PROJECT_NAME/derivatives/mrsi-<space>/sub-<subject_id>/ses-<session>/
   ```
- File naming convention :
  ```sh
  sub-<subject_id>_ses-<session>_space-<space>_met-<metabolite>_desc-<description>_mrsi.nii.gz
  ```

| **BIDS Prefix** | **Description**                           | **Choices**                                                                                           |
|-----------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `subject_id`         | Subject/Participant ID                    |                                                                                                       |
| `session`         | Session ID                                   | `[V1, V2, V3, ...]`                                                                                   |
| `space`       | MRI Acquisition space                     | `orig` or `origfilt` (for original MRSI space), `t1w`, `mni`                                                          |
| `metabolite`         | MRSI metabolite          | `Ins` myo-inositol <br>`CrPCr` creatine + phosphocreatine <br> `GPCPCh` lycerophosphocholine + phosphocholine   <br> `GluGln` glutamate + glutamine,   <br> `NAANAAG` N-acetylaspartate + N-acetylaspartylglutamate    <br>`water` water signal                                       |
| `description`        | Description of MRSI map                          |  `signal` MRSI signal <br> `crlb` (Cramer-Rao LowerBound LC model) <br> `fwhm` (full-width half-maximum LC model)  <br> `snr` (Signal/Noise LC model)  <br> `filtharmonic` (type of preprocessing filter)  <br> `brainmask` brain binary mask 

## Steps to Construct a within-subject MeSiM

1. **Create MRSI-to-T1w transforms**
   ```python
   python experiments/MeSiM_pipeline/registration_mrsi_to_t1.py --group Dummy-Project --ref_met CrPCr --subject_id S001 --session V1 --nthreads 16

2. **Map Chimera parcel image to MRSI space** 
   ```python
   python experiments/MeSiM_pipeline/map_parcel_image_to_mrsi.py --group Dummy-Project --subject_id S001 --session V1 --atlas LFMIHIFIS --scale 3

3. **Construct MeSiM** 
   ```python
   python experiments/MeSiM_pipeline/construct_MeSiM_subject.py --group Dummy-Project --subject_id S001 --session V1 --atlas LFMIHIFIS --scale 3 --npert 50 --show_plot 1 --nthreads 16 --analyze 1 

4. **Construct population averaged MeSiM from provided within-subject MeSiMs (Geneva-Study)** 
   ```python
    python experiments/MeSiM_pipeline/construct_MeSiM_pop.py --group Geneva-Study --scale 3 --parc LFMIHIFIS --participants experiments/MeSiM_pipeline/participant_list/best_Geneva-Study_sessions_mrsi.tsv

5. **Construct Metabolic Similarity Index (MSI) 3DMap from the population averaged MeSiM (Geneva-Study)** 
   ```python
    python experiments/MeSiM_pipeline/construct_MSI-map_pop.py --group Geneva-Study --scale 3 --parc LFMIHIFIS --dimalg pca_tsne --perplexity 30


- **Outputs**
    Warp transforms, coregistered Chimera parcellation label images, MeSiMs are saved in the ```/derivatives``` folder

### Description of input options

| **Arg Name**      | **Description**                                                                                                  | **Type**    | **Default**    |
|-------------------|------------------------------------------------------------------------------------------------------------------|-------------|---------------:|
| `--group`         | Name of the BIDS project folder or group to process.                                                           | string      | Dummy-Project  |
| `--subject_id`    | ID of the subject to process.<br>Example: sub-XX                                                                  | string      | S001           |
| `--session`       | Session ID.<br>Example: V1, V2, V3, ...                                                                           | string      | V1             |
| `--parc `         | Chimera parcellation string followed                    .                                                         | string      | LFMIHIFIS             |
| `--npert`         | Number of metabolic profile perturbations used.                                                                  | integer     | 50       |
| `--leave_one_out` | Add leave-one-metabolite-out option to MeSiM construction.<br>Accepts values: 0, 1.                                | int [0,1]   | 0              |
| `--show_plot`     | Display MeSiM and natural network analysis.<br>Accepts values: 0, 1.                                               | int [0,1]   | 0              |
| `--overwrite`     | Overwrite existing                                                                                               | int [0,1]     | 0              |
| `--ref_met`       | Reference MRSI metabolite.<br>Used as the moving image for coregistration.                                         | str         | CrPCr          |
| `--nthreads`      | Number of parallel CPU threads for processing.                                                                   | integer       | 4              |
| `--t1,        `    | T1w image file path                                                                                              | str           |None            |
| `--t1mask    `    | T1w brain mask image file path                                                                                   | str           |None            |


<!-- <img src="https://github.com/user-attachments/assets/4f0069ea-c4d7-4466-bd8e-7c55b1da3180" alt="Screenshot from 2025-03-11 22-40-35" width="600" /> -->

## Batch processing

1. **Create MRSI-to-T1w transforms**
   ```python
   python experiments/MeSiM_pipeline/registration_mrsi_to_t1_batch.py --group Dummy-Project --ref_met CrPCr --nthreads 16 --participants $PATH2_PARTICIPANT-SESSION_FILE --t1pattern acq-memprage_desc-brain_T1w

2. **Construct MeSiM** 
   ```python
   python experiments/MeSiM_pipeline/construct_MeSiM_subject_batch.py --group Dummy-Project --atlas LFMIHIFIS --scale 3 --npert 50 --show_plot 0 --nthreads 16 --analyze 1 --participants $PATH2_PARTICIPANT-SESSION_FILE --t1pattern acq-memprage_desc-brain_T1w


- **Notes**
     ```--participants``` list all the the participant and session ID  as defined by the standard BIDS ```participants_allsessions.tsv ```. If not specified ```--participants```  is set to ```$BIDSDATAPATH/group``` defined in the environment file ```.env``` created in step ```4. **Set up ENV paths**```
  

## Usefull MRSI processing tools

-  **Create T1w-to-MNI transforms (1 subject-session)**  
   ```python
   python experiments/MeSiM_pipeline/registration_t1_to_MNI.py --group Dummy-Project  --subject_id S001 --session V1 --nthreads 16 --t1 $PATH2TW_FILE

-  **Create T1w-to-MNI transforms (batch process)**  
   ```python
   python experiments/MeSiM_pipeline/registration_t1_to_MNI_batch.py --group Dummy-Project  --particpants $PATH2_PARTICIPANT-SESSION_FILE --nthreads 16 --t1pattern T1PATTERM


- **Coregister all MRSI metabolites to T1 & MNI space (1 subject-session)** 
   ```python
    python experiments/MeSiM_pipeline/transform_mrsi_to-t1_to-mni.py --group Dummy-Project --subject_id S001 --session V1  --nthreads 16

- **Coregister all MRSI metabolites to T1 & MNI space (batch process)** 
   ```python
    python experiments/MeSiM_pipeline/transform_mrsi_to-t1_to-mni_batch.py --group Dummy-Project  --particpants $PATH2_PARTICIPANT-SESSION_FILE  --nthreads 16


- **Notes**
     For batch processing, specify a T1 pattern  ```--t1pattern``` that matches the corresponding T1 image path (e.g ```acq-memprage```,```brain_T1W```)


