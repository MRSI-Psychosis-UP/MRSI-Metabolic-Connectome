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

3. **Coregister T1w to MNI sspace**  
   ```python
   experiments/registration_t1_to_MNI.py --group Dummy-Project --subject_id S001 --session V1 --nthreads 16

4. **Transform all MRSI metabolites** 
   ```python
   experiments/transform_mrsi_to-t1_to-mni.py --group Dummy-Project --subject_id S001 --session V1  --nthreads 16

5. **Parcellate MRSI with Chimera parcel image** 
   ```python
   experiments/parcellate_anatomical.py --group Dummy-Project --subject_id S001 --session V1

6. **Construct MRMeSiM** 
   ```python
   experiments/construct_MeSiM_subject.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --nthreads 16

7. **Construct Metabolic Similarity Map**
    ```python
    experiments/compute_MSI-map_subj.py --group Dummy-Project --subject_id S001 --session V1 --npert 50 --nthreads 16
