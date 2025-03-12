# Metabolic-Connectome

This repository contains all the necessary files to construct a within-subject metabolic similarity matrix (MeSiM) based on MRSI scans, as detailed in the following publication: [LINK 2ARXIV](http://link2arxiv).

## License
This project is licensed under the terms described in [LICENSE](./LICENSE).

## Datasets
A ```data/BIDS/Dummy-Project``` has been provided for demonstration purposes. For those interested in obtaining access to the full MRSI dataset beyond demonstration purposes, please reach out to the authors of this paper with a well-motivated request detailing your research aims and how the data will be used. 

## Install Toolbox

### Prerequisites

- **Python 3.x**: Ensure you have Python 3 installed on your system.
- **Conda or Miniconda**: Optionally, use Conda/Miniconda for environment management and package installation.



### Installation

1. **Clone the Repository**
   ```bash
   git clone git@github.com:MRSI-Psychosis-UP/Metabolic-Connectome.git
   cd Metabolic-Connectome
2. **Set up ENV paths**
   ```python
   python set_env_paths.py

 and navigate to your BIDS folder containing MRI data. Choose data/BIDS for demonstration purposes

3. **Install package**
    ```bash
    bash build_env.sh

### Steps to Construct within subjectMeSiM

1. **Create MRSI-to-T1w trasnforms**
   ```python
   python experiments/registration_mrsi_to_t1.py --group Dummy-Project --ref_met CrPCr --subject_id S001 --session V1 --nthreads 16

2. **Create T1w-to-MNI trasnforms**  
   ```python
   python experiments/registration_t1_to_MNI.py --group Dummy-Project  --subject_id S001 --session V1 --nthreads 16


3. **Parcellate MRSI with Chimera parcel image** 
   ```python
   python experiments/parcellate_anatomical.py --group Dummy-Project --subject_id S001 --session V1 --atlas LFMIHIFIF-3

4. **Construct MeSiM** 
   ```python
   python experiments/construct_MeSiM_subject.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --show_plot 1 --nthreads 16


- **Outputs**
    Warp transforms, coregistered Chimera parcellation label images, MeSiMs are save in the ```/derivatives``` folder

## Description of Arguments

| **Arg Name**     | **Description**                                           | **Type**  | **Default**    |
|------------------|-----------------------------------------------------------|-----------|---------------:|
| `--group`        | Name of the BIDS project folder or group to process.      | string    | Dummy-Project  |
| `--subject_id`   | ID of the subject to process. sub-XX                      | string    | S001           |
| `--session`      | Session ID [V1,V2,V3,...]                                 | string    | V1             |
| `--nthreads`     | Number of parallel CPU threads for processing.            | integer   | 4              |
| `--atlas`        | Chimera parcellation string followed by scale parameter   | string    | 50             |
| `--npert`        | Number of metabolic profile perturbations used            | integer   | 50             |
| `--leave_one_out`| Add leave-one-metabolite-out option to MeSiM construction | int [0,1] | 0              |
| `--show_plot`    | Display MeSiM and natural network analyis                 | int [0,1] | 0              |
| `--ref_met`      | Reference MRSI metabolite, used as the moving image for coregistration          | str   | CrPCr             |

<!-- <img src="https://github.com/user-attachments/assets/4f0069ea-c4d7-4466-bd8e-7c55b1da3180" alt="Screenshot from 2025-03-11 22-40-35" width="600" /> -->
![](figures/Figure1.png)


### Usefull MRSI preprocessing tools

- **Coregister all MRSI metabolites to T1 & MNI space** 
   ```python
    python experiments/transform_mrsi_to-t1_to-mni.py --group Dummy-Project --subject_id S001 --session V1  --nthreads 16


- **Construct Metabolic Similarity Map**
    ```python
    python experiments/compute_MSI-map_subj.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --nthreads 16


