# 🧠 MRSI Toolbox Kit

This repository provides tools and preprocessing utilites to construct a within-subject **Metabolic Similarity Matrix (MeSiM)** based on MRSI scans, as detailed in [biorxiv](https://www.biorxiv.org/content/10.1101/2025.03.10.642332v1) and preparing files for a voxel-based analysis as detailed in [biorxiv](https://www.biorxiv.org/content/10.1101/2025.06.22.660965v1)

## 📚 Table of Contents

- [🧩 Construct a within-subject MeSiM](#-construct-a-within-subject-mesim)
- [📊 MeSiM Analysis](#-mesim-analysis)
- [🔧 Pre-Processing Pipeline for Voxel-Based Analysis](#-pre-processing-pipeline-for-voxel-based-analysis)


![Figure 1](figures/Figure1.png)

---

## 📜 License

Licensed under the terms specified in [LICENSE](./LICENSE).

---

## 🧑‍💻 Contributors

| Name               | GitHub Profile                                   | Email                      |
|--------------------|--------------------------------------------------|----------------------------|
| Federico Lucchetti | [@fedlucchetti](https://github.com/fedlucchetti) | federico.lucchetti@unil.ch |
| Edgar Céléreau     | [@bobdev](https://github.com/ecelreau)           | edgar.celereau@unil.ch     |

---
## 📂 Dataset

A demo dataset is available at `data/BIDS/Dummy-Project` and constructed MeSiMs from the Geneva-Study in `data/BIDS/Geneva-Study/derivatives/connectivity`.

To access the full dataset, contact the authors with a detailed research proposal explaining your intended use.

---

## ⚙️ Installation

### Requirements

- **Python 3.x**
- **Conda / Miniconda** (optional, but recommended)
- **[CHIMERA](https://github.com/connectomicslab/chimera)** for anatomical parcellation

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone git@github.com:MRSI-Psychosis-UP/MRSI-Metabolic-Connectome.git
   cd MRSI-Metabolic-Connectome
   ```

2. **Install the Environment**
   ```bash
   bash build_env.sh
   ```

3. **Activate the Environment**
   ```bash
   conda activate mrsitooldemo_env
   ```

4. **Set Environment Paths**
   ```bash
   python set_env_paths.py
   ```
   Use the provided demo BIDS dataset (`data/BIDS`) if applicable.

---

## 🗂️ Inputs

### List of Participants/Subject File
- BIDS directory `PROJECT_NAME/` should contain a tab-separated `participants_allsessions.tsv` file following the BIDS standard `subject-id \t session-id`.

### MRSI Files

- MRSI files should be placed in:
   ```
   PROJECT_NAME/derivatives/mrsi-<space>/sub-<subject_id>/ses-<session>/
   ```

- File naming convention:
   ```
   sub-<subject_id>_ses-<session>_space-<space>_met-<metabolite>_desc-<description>_mrsi.nii.gz
   ```

| **BIDS Prefix**  | **Description**           | **Choices**                                                                                           |
|------------------|---------------------------|-------------------------------------------------------------------------------------------------------|
| `subject_id`     | Subject/Participant ID    |                                                                                                       |
| `session`        | Session ID                | `[V1, V2, V3, ...]`                                                                                   |
| `space`          | MRI Acquisition space     | `orig`, `t1w`, `mni`                                                                                  |
| `metabolite`     | MRSI resolved Metabolite  | **B<sub>0</sub> = 3T**: `Ins`, `CrPCr`, `GPCPCh`, `GluGln`, `NAANAAG`, `water`                                        |
|                  |                           | **B<sub>0</sub> = 7T**: `NAA`, `NAAG`, `Ins`, `GPCPCh`, `Glu`, `Gln`, `CrPCr`, `GABA`, `GSH`                          |
| `description`    | MRSI Map Description      | `signal`, `crlb`, `fwhm`, `snr`, `filtharmonic`, `brainmask`                                          |
### Anatomical Files

- **Chimera Anatomical Parcellation Files:**
   - Example for all subjects found in `Dummy-Project`:
      ```bash
      chimera -b data/BIDS/Dummy-Project/ \
              -d data/BIDS/Dummy-Project/derivatives/ \
              --freesurferdir data/BIDS/Dummy-Project/derivatives/freesurfer/ \
              -p LFMFIIFIS -g 2
      ```
   - Parcellations saved in `PROJECT_NAME/derivatives/chimera-atlases`.

- **Partial Volume Correction (PVC) files:**

  Replace `<N>` with the appropriate tissue type index (e.g., `1` for GM, `2` for WM, `3` for CSF):

   ```bash
   PROJECT_NAME/derivatives/<PVCORR_DIR>/sub-<subject_id>/ses-<session>/sub-<subject_id>_ses-<session>_desc-p<N>_T1w.nii.gz
   ```

  - **Minimum required:**
    - `p1`: gray matter
    - `p2`: white matter
    - `p3`: CSF

  - **Additional tissue files may be included (optional).**

---

## 🧩 Construct a within-subject MeSiM

1. **Create MRSI-to-T1w Transforms**
   ```bash
   python experiments/Preprocessing/registration_mrsi_to_t1.py --group Dummy-Project --ref_met CrPCr --subject_id S001 --session V1 --nthreads 16
   ```

2. **Map Chimera Parcel Image to MRSI Space**
   ```bash
   python experiments/MeSiM_pipeline/map_parcel_image_to_mrsi.py --group Dummy-Project --subject_id S001 --session V1 --parc LFMIHIFIS --scale 3
   ```

3. **Construct within-subject MeSiM**
   ```bash
   python experiments/MeSiM_pipeline/construct_MeSiM_subject.py --group Dummy-Project --subject_id S001 --session V1 --parc LFMIHIFIS --scale 3 --npert 50 --show_plot 1 --nthreads 16 --analyze 1
   ```

4. **Construct within-subject MeSiM (batch)**
   ```bash
   python experiments/MeSiM_pipeline/construct_MeSiM_subject_batch.py --group Dummy-Project --parc LFMIHIFIS --scale 3 --npert 50 --show_plot 0 --nthreads 16 --analyze 1 --participants $PATH2_PARTICIPANT-SESSION_FILE --t1pattern acq-memprage_desc-brain_T1w
   ```

5. **Construct MeSiM Population Average**
   ```bash
   python experiments/MeSiM_pipeline/construct_MeSiM_pop.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --participants $PATH2_PARTICIPANT-SESSION_FILE

- **Outputs**: Transforms, coregistered parcellations, and MeSiMs are saved in the `derivatives/` folder.

---

### Input Options Description

| **Arg Name**      | **Description**                                                                 | **Type**      | **Default**    |
|-------------------|---------------------------------------------------------------------------------|---------------|----------------|
| `--group`         | BIDS project folder name                                                        | string        | Dummy-Project  |
| `--subject_id`    | Subject ID, e.g., S001                                                          | string        | S001           |
| `--session`       | Session ID, e.g., V1                                                            | string        | V1             |
| `--parc`          | Chimera parcellation string                                                     | string        | LFMIHIFIS      |
| `--npert`         | Number of metabolic profile perturbations                                       | integer       | 50             |
| `--leave_one_out` | Leave-one-metabolite-out option (0 or 1)                                        | int [0,1]     | 0              |
| `--show_plot`     | Show plots (0 or 1)                                                             | int [0,1]     | 0              |
| `--overwrite`     | Overwrite existing results (0 or 1)                                             | int [0,1]     | 0              |
| `--ref_met`       | Reference metabolite for coregistration                                         | string        | CrPCr          |
| `--nthreads`      | Number of parallel CPU threads                                                  | integer       | 4              |
| `--t1`            | Path to T1-weighted image                                                       | string        | None           |
| `--t1mask`        | Path to T1-weighted brain mask                                                  | string        | None           |
| `--b0`            | MRI  B<sub>0</sub> field in Tesla (3 or 7)                                      | float         | 3              |


- **Note**: `--participants` refers to a BIDS-style `participants_allsessions.tsv`. Defaults to `$BIDSDATAPATH/group` if not specified.
---




---

## 📊 MeSiM Analysis

1. **Construct Metabolic Similarity Map (Single Subject)**
   ```bash
   python experiments/MeSiM_analysis/construct_MSI-map_subj.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --dimalg pca_tsne
   ```

2. **Construct Metabolic Similarity Map (Population)**
   ```bash
   python experiments/MeSiM_analysis/construct_MSI-map_pop.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --dimalg pca_tsne --msiscale -255.0
   ```

3. **Inverse Map MSI to MRSI Signal (Population)**
   ```bash
   python experiments/MeSiM_analysis/construct_MSI-map_pop.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --dimalg pca_tsne
   ```

4. **Derive all GM constrained networtk paths**
   ```bash
   python experiments/MeSiM_analysis/find_all_network_paths.py --group Geneva-Study --parc LFMIHIFIS --scale 3
   --lobe LOBE --hemi HEMI --lpath 13
   ```
5. **Construct Metabolic Principal Curve (Population)**

   ```bash
   python experiments/MeSiM_analysis/construct_metabolic_principal_path.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --diag patients --lpath 13 --group LPN-Project --lobe ctx --nperm 100 --lobe ctx
   ```
- **Note**:  
  - Run first `find_all_network_paths.py` for both hemispheres `lh` and `rh` to construct all possible network paths then select the one which maximizes metabolic entropy and minimizes local metabolic heterogeneity with `construct_metabolic_principal_path.py` followed by comparison with a random geometric network and results figure generation.
  - `--dimalg` specifies the **manifold-discovery algorithm** used to construct the metabolic fibre.  
  - `--hemi`  chooses the **hemisphere** in which the fibre is built (`lh` or `rh`).  
  - `--lpath` sets the **maximum path length**.  
  - `--nperm` defines the **size of the null distribution**, generated from random-geometric networks.  
  - `--start` and `--stop` indicate the **start and stop nodes**.  
    - If not provided, they default to the **occipital region** (start) and **frontal/anterior cingulate regions** (stop), which maximise the inter-node MS-mode difference.  
    - The script then runs `find_all_network_paths.py` from scratch with these adjusted end-node labels.  
  - `--lobe` restricts the path search to either the **neocortex** (`ctx`) or the **subcortex** (`subc`).
---

![Metabolic Fibre](figures/metab_fibre.png)


## 🔧 Pre-Processing Pipeline for Voxel-Based Analysis

> **Batch mode semantics**
>
> * `--batch file` **requires** `--participants` to point to a `.tsv` file listing the subject–session pairs to process.
> * `--batch off` **requires** both `--subject_id` and `--session` and processes **one** acquisition (a single subject–session pair).
> * `--batch all` processes all discoverable subject–session pairs in the group.

* **Create MRSI-to-T1w Transforms (batch)**

  ```bash
  python experiments/Preprocess/registration_mrsi_to_t1.py \
    --group Dummy-Project --ref_met CrPCr --nthreads 16 \
    --batch file --participants $PATH2_PARTICIPANT-SESSION_FILE \
    --t1 acq-memprage_desc-brain_T1w
  ```

* **Create T1w-to-MNI Transforms (batch)**

  ```bash
  python experiments/Preprocess/registration_t1_to_MNI.py \
    --group Dummy-Project --nthreads 16 \
    --batch file --participants $PATH2_PARTICIPANT-SESSION_FILE
  ```

* **Preprocess All MRSI Metabolites to T1 & MNI + partial volume correction (single subject)**

  ```bash
  python experiments/Preprocess/preprocess.py \
    --group Dummy-Project --nthreads 16 --b0 3 --overwrite 1 \
    --batch off --subject_id S001 --session V1
  ```

* **Preprocess All MRSI Metabolites to T1 & MNI + partial volume correction (batch)**

  ```bash
  python experiments/Preprocess/preprocess.py \
    --group Dummy-Project --nthreads 16 --b0 3 --overwrite 1 \
    --batch file --participants $PATH2_PARTICIPANT-SESSION_FILE
  ```

* **Create population quality mask**

  ```bash
  python experiments/Preprocess/compute_pop_qmask.py \
    --group Dummy-Project --participants $PATH2_PARTICIPANT-SESSION_FILE \
    --snr 4 --crlb 20 --fwhm 0.1 --alpha 0.68 --b0 3
  ```

### Command-Line Arguments

| Argument         | Type  | Default                               | Description                                                                                                                                                     |
| ---------------- | ----- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--group`        | str   | `Dummy-Project`                       | Name of the group/study directory.                                                                                                                              |
| `--overwrite`    | int   | `0` (choices: `0`, `1`)               | Overwrite existing outputs; set to `1` to enable.                                                                                                               |
| `--batch`        | str   | `off` (choices: `all`, `file`, `off`) | Batch mode. `file` **requires** `--participants`; `off` **requires** `--subject_id` **and** `--session` (processes a single acquisition); `all` uses all pairs. |
| `--participants` | path  | `None`                                | Path to a `.tsv` listing subject–session pairs (**required when** `--batch file`; ignored for `--batch off` and `--batch all`).                                 |
| `--subject_id`   | str   | `None`                                | Subject identifier (**required when** `--batch off`; ignored otherwise).                                                                                        |
| `--session`      | str   | `None`                                | Session label (**required when** `--batch off`; ignored otherwise).                                                                                             |
| `--snr`          | float | `4`                                   | SNR threshold above which an MRSI signal is considered significant (per voxel).                                                                                 |
| `--crlb`         | float | `20`                                  | CRLB threshold below which an MRSI signal is considered significant (per voxel).                                                                                |
| `--fwhm`         | float | `0.1`                                 | FWHM threshold below which an MRSI signal is considered significant (per voxel).                                                                                |
| `--alpha`        | float | `0.68`                                | Proportion of significant voxels across participants required to retain a voxel at the group level.                                                             |
| `--b0`           | float | `3` (choices: `3`, `7`)               | MRI B0 field strength in Tesla.                                                                                                                                 |
                                                                                                |


![Figure 1](figures/preproc_vba.png)









