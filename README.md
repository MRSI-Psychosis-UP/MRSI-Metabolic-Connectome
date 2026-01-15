# 🧠 MRSI Toolbox Kit

This repository provides tools and preprocessing utilites to construct a within-subject **Metabolic Similarity Matrix (MetSiM)** based on MRSI scans, as detailed in [Nature Communications 2025](https://www.nature.com/articles/s41467-025-66124-w) and preparing files for a voxel-based analysis as detailed in [biorxiv](https://www.biorxiv.org/content/10.1101/2025.06.22.660965v1)

## 📚 Table of Contents

- [🧩 Construct a within-subject MetSiM](#-construct-a-within-subject-metsim)
- [📊 MetSiM Analysis](#-metsim-analysis)
- [🔧 Pre-Processing Pipeline for Voxel-Based Analysis](#-pre-processing-pipeline-for-voxel-based-analysis)

---

## 📜 License

The repository is distributed under the CHUV license [LICENSE](./LICENSE).

---

## 🧑‍💻 Contributors

| Name               | GitHub Profile                                   | Email                      |
|--------------------|--------------------------------------------------|----------------------------|
| Federico Lucchetti | [@fedlucchetti](https://github.com/fedlucchetti) | federico.lucchetti@unil.ch |
| Edgar Céléreau     | [@mrspsy](https://github.com/mrspsy)             | edgar.celereau@unil.ch     |

---
## 📂 Dataset

A demo dataset is available at `data/BIDS/Dummy-Project` and constructed MetSiMs from the Geneva-Study in `data/BIDS/Geneva-Study/derivatives/connectivity`.

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
   - Example for subjects T1w filenames stores in found in `t1s.txt` with chimera atlas LFMIHISIFF using the Lausanne cortical parcellation at scale 3 :
      ```bash
      chimera -b data/BIDS/Dummy-Project/ \
              -d data/BIDS/Dummy-Project/derivatives/ \
              --freesurferdir data/BIDS/Dummy-Project/derivatives/freesurfer/ \
              -p LFMIHISIFF -g 2 -s 3 -ids t1s.txt --nthreads 28
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

## 🧩 Construct a within-subject MetSiM

> **Batch mode semantics**
>
> * `--batch file` **requires** `--participants` to point to a `.tsv` file listing the subject–session pairs to process.
> * `--batch off` **requires** both `--subject_id` and `--session` and processes **one** acquisition (a single subject–session pair).
> * `--batch all` processes all discoverable subject–session pairs in the group.


1. **Create MRSI-to-T1w Transforms**
   ```bash
   python experiments/Preprocessing/registration_mrsi_to_t1.py --group Dummy-Project --ref_met CrPCr --subject_id S001 --session V1 --nthreads 16
   ```

2. **Map Chimera Parcel Image to MRSI Space**
   ```bash
   python experiments/MetSiM_pipeline/map_parcel_image_to_mrsi.py --group Dummy-Project --subject_id S001 --session V1 --parc LFMIHIFIS --scale 3
   ```

3. **Construct within-subject MetSiM**
   ```bash
   python experiments/MetSiM_pipeline/construct_MetSiM_subject.py --group Dummy-Project --subject_id S001 --session V1 --parc LFMIHIFIS --scale 3 --npert 50 --show_plot 1 --nthreads 16 --analyze 1
   ```

4. **Construct within-subject MetSiM (batch)**
   ```bash
   python experiments/MetSiM_pipeline/construct_MetSiM_subject.py --group Dummy-Project --parc LFMIHIFIS --scale 3 --npert 50 --show_plot 0 --nthreads 16 --analyze 1 --batch file --participants $PATH2_PARTICIPANT-SESSION_FILE --t1mask acq-memprage_desc-brain_T1w 
   ```

5. **Construct MetSiM Population Average**
   ```bash
   python experiments/MetSiM_pipeline/construct_MetSiM_pop.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --participants $PATH2_PARTICIPANT-SESSION_FILE

- **Outputs**: Transforms, coregistered parcellations, and MetSiMs are saved in the `derivatives/` folder.

---

### Input Options Description

| **Arg Name**      | **Description**                                                                                                                                                 | **Type**                          | **Default**     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | --------------- |
| `--group`         | BIDS project folder name                                                                                                                                        | str                               | `Dummy-Project` |
| `--subject_id`    | Subject ID (e.g., `S001`). **Required when** `--batch off` (processes a single acquisition).                                                                    | str                               | `S001`          |
| `--session`       | Session label (e.g., `V1`). **Required when** `--batch off` (processes a single acquisition).                                                                   | str                               | `V1`            |
| `--parc`          | Chimera parcellation string                                                                                                                                     | str                               | `LFMIHIFIS`     |
| `--npert`         | Number of metabolic profile perturbations                                                                                                                       | int                               | `50`            |
| `--leave_one_out` | Leave-one-metabolite-out option                                                                                                                                 | int (0 or 1)                      | `0`             |
| `--show_plot`     | Show plots                                                                                                                                                      | int (0 or 1)                      | `0`             |
| `--overwrite`     | Overwrite existing results                                                                                                                                      | int (0 or 1)                      | `0`             |
| `--ref_met`       | Reference metabolite for coregistration                                                                                                                         | str                               | `CrPCr`         |
| `--nthreads`      | Number of parallel CPU threads                                                                                                                                  | int                               | `4`             |
| `--t1`            | Path or pattern to T1-weighted image                                                                                                                            | str                               | `None`          |
| `--t1mask`        | Path or pattern to T1-weighted brain mask                                                                                                                       | str                               | `None`          |
| `--b0`            | MRI B<sub>0</sub> field in Tesla                                                                                                                                | float (choices: 3, 7)             | `3`             |
| `--batch`         | Batch mode. `file` **requires** `--participants`; `off` **requires** `--subject_id` **and** `--session` (processes a single acquisition); `all` uses all pairs. | str (choices: `all`,`file`,`off`) | `off`           |
| `--participants`  | Path to a `.tsv` listing subject–session pairs to process (**required when** `--batch file`; ignored for `--batch off` and `--batch all`).                      | path                              | `None`          |

- **Note**: `--participants` refers to a BIDS-style `participants_allsessions.tsv`. Defaults to `$BIDSDATAPATH/group` if not specified.
---


![Figure 1](figures/Figure1.png)

---

## 📊 MetSiM Analysis

1. **Construct Metabolic Similarity Map (Single Subject)**
   ```bash
   python experiments/MetSiM_analysis/construct_MSI-map_subj.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --dimalg pca_tsne
   ```

2. **Construct Metabolic Similarity Map (Population)**
   ```bash
   python experiments/MetSiM_analysis/construct_MSI-map_pop.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --dimalg pca_tsne --msiscale -255.0
   ```

3. **Inverse Map MSI to MRSI Signal (Population)**
   ```bash
   python experiments/MetSiM_analysis/construct_MSI-map_pop.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --npert 50 --dimalg pca_tsne
   ```

4. **Derive all GM constrained networtk paths**
   ```bash
   python experiments/MetSiM_analysis/find_all_network_paths.py --group Geneva-Study --parc LFMIHIFIS --scale 3
   --lobe LOBE --hemi HEMI --lpath 13
   ```
5. **Construct Metabolic Principal Curve (Population)**

   ```bash
   python experiments/MetSiM_analysis/construct_metabolic_principal_path.py --group Geneva-Study --parc LFMIHIFIS --scale 3 --diag group --lpath 13 --lobe ctx --nperm 100 --lobe ctx
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

`experiments/Preprocessing/preprocess.py` now runs the full voxel-wise chain end-to-end: preflight checks, optional orientation correction, spike filtering, optional partial volume correction, and MRSI exports to MNI or T1w space. Missing transforms are generated where supported; overwrite flags force regeneration.

**What it does**
- Checks required inputs (T1w, MRSI signals/CRLB/SNR/FWHM, CAT12 p1–p3 where available) and batches subject–session pairs from `participants_allsessions.tsv` or a custom TSV/CSV.
- Prints a preflight availability table per subject/session: existing files are marked with a green check, missing ones with a red X, and items marked **PROC** (orange) are auto-generated during preprocessing (e.g., transforms or masks).
- Optional oblique FOV correction (`--corr_orient`).
- Filters MRSI spikes (`--filtoption`, `--spikepc`) and builds brain masks if present.
- Runs/refreshes T1w→MNI registration when needed (`--overwrite_mni_reg`) and exports MRSI to T1w or MNI space at native-MRSI or T1w resolution via `--transform`.
- Partial volume correction runs when `--overwrite_pve` is set and p1/p2/p3 maps are present; otherwise PVC is skipped (or explicitly bypassed with `--no_pvc`).

> **Batch mode semantics**
>
> * `--batch file` **requires** `--participants` to point to a `.tsv`/`.csv` listing the subject–session pairs to process.
> * `--batch off` **requires** both `--sub` and `--ses` and processes **one** acquisition.
> * `--batch all` processes all discoverable subject–session pairs in the group.

**Quick start**

- Single subject (filtering → PVC → export to MNI @ native MRSI res)
  ```bash
  python experiments/Preprocessing/preprocess.py \
    --group Dummy-Project --sub S001 --ses V1 \
    --t1 acq-memprage_desc-brain_T1w --b0 3 --nthreads 16 \
    --transform mni-origres --overwrite_pve
  ```
- Batch (participants TSV/CSV)
  ```bash
  python experiments/Preprocessing/preprocess.py \
    --group Dummy-Project --b0 3 --nthreads 16 \
    --batch file --participants $PATH2_PARTICIPANT-SESSION_FILE \
    --t1 acq-memprage_desc-brain_T1w \
    --transform mni-t1wres --overwrite_pve
  ```
- Optional add-ons: `--corr_orient` to fix oblique FOV; `--no_pvc` to skip partial volume correction; `--overwrite_filt`/`--overwrite_transform`/`--overwrite_mni_reg` to recompute stages; `--proc_mnilong` (+ `--overwrite_mnilong`) for MNI152-long outputs; `--checksum` to print a pre-run output validity summary; `--v 1` for verbose logs.

**Key `preprocess.py` options**

| Argument | Default | Purpose |
| -------- | ------- | ------- |
| `--group` | `Mindfulness-Project` | BIDS project under `$BIDSDATAPATH`. |
| `--sub` | `S002` | Subject ID used when `--batch off`. |
| `--ses` | `V3` | Session ID used when `--batch off`. |
| `--batch` | `off` (`all`/`file`/`off`) | Process all pairs, a TSV/CSV list (`--participants`), or a single pair. |
| `--participants` | `None` | TSV/CSV path used when `--batch file` (columns like `participant_id`/`sub` and `ses`/`session_id`). |
| `--t1` | `desc-brain_T1w` | T1w path or pattern (resolved per subject/session). |
| `--b0` | `3` (`3`/`7`) | Sets metabolite list (3T vs 7T). |
| `--nthreads` | `4` | CPU threads for filtering, PVC, and transforms. |
| `--filtoption` | `filtbiharmonic` | Spike filtering strategy. |
| `--spikepc` | `99` | Percentile for spike removal. |
| `--transform` | `mni-origres` (`mni-t1wres`/`t1w-origres`/`t1w-t1wres`) | Export MRSI to MNI or T1w space at native MRSI vs T1w resolution. |
| `--overwrite_transform` | `off` | Recompute transforms/exports for the selected space+resolution (flag). |
| `--no_pvc` | `off` | Skip PVC exports (transform filtered signals only; flag). |
| `--overwrite_pve` | `off` | Run/refresh partial volume correction before transforms (flag). |
| `--overwrite_filt` | `off` | Recompute spike filtering outputs (flag). |
| `--overwrite_t1_reg` | `off` | Force regeneration of MRSI→T1w transforms (flag). |
| `--overwrite_mni_reg` | `off` | Force regeneration of T1w→MNI transforms (flag). |
| `--proc_mnilong` | `off` | Generate MNI152-longitudinal outputs (flag). |
| `--overwrite_mnilong` | `off` | Rerun MNI152-longitudinal exports (flag). |
| `--corr_orient` | `off` | Correct oblique FOV orientation for MRSI and masks (flag). |
| `--checksum` | `off` | Display the output validity summary before processing (flag). |
| `--v` | `0` | Verbose flag (`1` to print more detail). |

Input notes: `--participants` can be TSV/CSV (subject column such as `participant_id`/`sub`/`id` and session column such as `ses`/`session_id`); default is `$BIDSDATAPATH/<group>/participants_allsessions.tsv` (V2BIS rows are skipped). `--t1` is required (defaults to the `desc-brain_T1w` pattern). Boolean options are flags (no `0/1` values); include them to enable. PVC needs CAT12 p1/p2/p3 maps; if absent or `--no_pvc` is set, PVC is skipped and logged.

**Population quality mask**

```bash
python experiments/Preprocessing/compute_pop_qmask.py \
  --group Dummy-Project --participants $PATH2_PARTICIPANT-SESSION_FILE \
  --snr 4 --crlb 20 --fwhm 0.1 --alpha 0.68 --b0 3
```

**Registration helpers (optional)**  
All registration is triggered automatically by `preprocess.py`. Run these directly only for debugging or bespoke registration:

- MRSI→T1w (batch capable)
  ```bash
  python experiments/Preprocessing/registration_mrsi_to_t1.py \
    --group Dummy-Project --ref_met CrPCr --nthreads 16 \
    --batch file --participants $PATH2_PARTICIPANT-SESSION_FILE \
    --t1 acq-memprage_desc-brain_T1w
  ```
- T1w→MNI (batch capable)
  ```bash
  python experiments/Preprocessing/registration_t1_to_MNI.py \
    --group Dummy-Project --nthreads 16 \
    --batch file --participants $PATH2_PARTICIPANT-SESSION_FILE
  ```


![Figure 1](figures/preproc_vba.png)



