# Metabolic-Connectome

This repository contains all the necessary files to construct a within-subject metabolic similarity matrix (MeSiM) based on MRSI scans, as detailed in the following publication: [LINK 2ARXIV](http://link2arxiv).

## Install Toolbox

### Prerequisites

- **Python 3.x**: Ensure you have Python 3 installed on your system.
- **Conda or Miniconda**: Optionally, use Conda/Miniconda for environment management and package installation.

### Datasets
A `data/BIDS` has been provided for demonstration purposes. For those interested in obtaining access to the full MRSI dataset beyond demonstration purposes, please reach out to the authors of this paper with a well-motivated request detailing your research aims and how the data will be used. 

### Installation

1. **Clone the Repository**
   ```bash
   git clone git@github.com:MRSI-Psychosis-UP/Metabolic-Connectome.git
   cd Metabolic-Connectome
2. **Set up ENV paths**
   ```python
   set_env_paths.py

 and navigate to your BIDS folder containing MRI data.  Choose data/BIDS for demonsration purposes

3. **Install package**
    ```bash
    build_env.sh

### Steps to Construct within subjectMeSiM

1. **Coregister MRSI to T1w**
   ```python
   experiments/registration_mrsi_to_t1.py --group Dummy-Project --subject_id S001 --session V1 --nthreads 16

2. **Coregister T1w to MNI sspace**  
   ```python
   experiments/registration_t1_to_MNI.py --group Dummy-Project --subject_id S001 --session V1 --nthreads 16
3. **Transform all MRSI metabolites** 
   ```python
   experiments/transform_mrsi_to-t1_to-mni.py --group Dummy-Project --subject_id S001 --session V1  --nthreads 16

4. **Parcellate MRSI with Chimera parcel image** 
   ```python
   experiments/parcellate_anatomical.py --group Dummy-Project --subject_id S001 --session V1

5. **Construct MeSiM** 
   ```python
   experiments/construct_MeSiM_subject.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --nthreads 16

6. **Construct Metabolic Similarity Map**
    ```python
    experiments/compute_MSI-map_subj.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --nthreads 16

## Description of Arguments

| **Arg Name**   | **Description**                                         | **Type**  | **Default** |
|----------------|---------------------------------------------------------|-----------|------------:|
| `--group`      | Name of the BIDS project folder or group to process.    | string    | None        |
| `--subject_id` | ID of the subject to process. sub-XX                    | string    | None        |
| `--session`    | Session ID [V1,V2,V3,...]                               | string    | None        |
| `--nthreads`   | Number of parallel CPU threads for processing.          | integer   | 4           |
| `--npert`      | Number of metabolic profile perturbations used          | integer   | 50          |

![Screenshot from 2025-03-11 22-40-35](https://github.com/user-attachments/assets/4f0069ea-c4d7-4466-bd8e-7c55b1da3180)




