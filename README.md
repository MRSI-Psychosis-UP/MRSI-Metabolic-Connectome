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
           -p LFMFIIFIFF -g 2
   ```
- Use ```-p LFMFIIFIFF -g 2``` supra-region sequence with 2mm GM grow inside WM in order to replicate the results in the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.10.642332v1).
- Chimera outputs are saved  ```PROJECT_NAME/derivatives/chimera-atlases```

### MRSI files ###
- Reconstrcuted MRSI files should be placed in the following derivatives folder
   ```sh
   PROJECT_NAME/derivatives/mrsi-<space>/sub-<subject_id>/ses-<session>/
   ```
- File naming convention :
  ```sh
  sub-<subject_id>_ses-<session>_space-<space>_acq-<acq>_desc-<metabolite>_mrsi.nii.gz
  ```

| **BIDS Prefix** | **Description**                           | **Choices**                                                                                           |
|-----------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `subject_id`         | Subject/Participant ID                    |                                                                                                       |
| `session`         | Session                                   | `[V1, V2, V3, ...]`                                                                                   |
| `space`       | MRI Acquisition space                     | `orig` or `origfilt` (for original MRSI space), `t1w`, `mni`                                                          |
| `acq`         | Type of reconstructed MRSI map           | `conc` (metabolite signal)<br>`crlb` (LCmodel-issued CRLB map)                                          |
| `metabolite`        | Metabolite string                         | `Ins` myo-inositol <br>`CrPCr` creatine + phosphocreatine <br> `GPCPCh` lycerophosphocholine + phosphocholine   <br> `GluGln` glutamate + glutamine,   <br> `NAANAAG` N-acetylaspartate + N-acetylaspartylglutamate 

## Steps to Construct a within-subject MeSiM

1. **Create MRSI-to-T1w transforms**
   ```python
   python scripts/registration_mrsi_to_t1.py --group Dummy-Project --ref_met CrPCr --subject_id S001 --session V1 --nthreads 16

2. **Create T1w-to-MNI transforms**  
   ```python
   python scripts/registration_t1_to_MNI.py --group Dummy-Project  --subject_id S001 --session V1 --nthreads 16


3. **Parcellate MRSI with Chimera parcel image** 
   ```python
   python scripts/parcellate_anatomical.py --group Dummy-Project --subject_id S001 --session V1 --atlas LFMIHIFIF-3

4. **Construct MeSiM** 
   ```python
   python scripts/construct_MeSiM_subject.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --show_plot 1 --nthreads 16

- **Outputs**
    Warp transforms, coregistered Chimera parcellation label images, MeSiMs are saved in the ```/derivatives``` folder

### Description of input options

| **Arg Name**      | **Description**                                                                                                  | **Type**    | **Default**    |
|-------------------|------------------------------------------------------------------------------------------------------------------|-------------|---------------:|
| `--group`         | Name of the BIDS project folder or group to process.                                                           | string      | Dummy-Project  |
| `--subject_id`    | ID of the subject to process.<br>Example: sub-XX                                                                  | string      | S001           |
| `--session`       | Session ID.<br>Example: V1, V2, V3, ...                                                                           | string      | V1             |
| `--atlas`         | Chimera parcellation string followed by scale parameter.                                                         | string      | 50             |
| `--npert`         | Number of metabolic profile perturbations used.                                                                  | integer     | 50             |
| `--leave_one_out` | Add leave-one-metabolite-out option to MeSiM construction.<br>Accepts values: 0, 1.                                | int [0,1]   | 0              |
| `--show_plot`     | Display MeSiM and natural network analysis.<br>Accepts values: 0, 1.                                               | int [0,1]   | 0              |
| `--overwrite`     | Overwrite existing                                                                                               | int [0,1]   | 0              |
| `--ref_met`       | Reference MRSI metabolite.<br>Used as the moving image for coregistration.                                         | str         | CrPCr          |
| `--nthreads`      | Number of parallel CPU threads for processing.                                                                   | integer     | 4              |
| `--t1_pattern`    | T1w image file pattern used for image registration                                                                 | str     | _run-01_acq-memprage_              |


<!-- <img src="https://github.com/user-attachments/assets/4f0069ea-c4d7-4466-bd8e-7c55b1da3180" alt="Screenshot from 2025-03-11 22-40-35" width="600" /> -->


## Usefull MRSI preprocessing tools

- **Coregister all MRSI metabolites to T1 & MNI space** 
   ```python
    python scripts/transform_mrsi_to-t1_to-mni.py --group Dummy-Project --subject_id S001 --session V1  --nthreads 16


- **Construct Metabolic Similarity Map**
    ```python
    python scripts/compute_MSI-map_subj.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --nthreads 16


