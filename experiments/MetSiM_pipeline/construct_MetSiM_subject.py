import os, sys, copy, shutil, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from graphplot.simmatrix import SimMatrixPlot
import pandas as pd
import nibabel as nib
from tools.datautils import DataUtils
from os.path import split, join, exists, isdir
from tools.filetools import FileTools
from tools.debug import Debug
from connectomics.parcellate import Parcellate
from randomize.randomize import Randomize
from connectomics.network import NetBasedAnalysis
from connectomics.nettools import NetTools
from connectomics.mesim import MeSiM
from registration.registration import Registration
from tools.mridata import MRIData
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

METABOLITES           = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
PARC_CEREB_SCHEME     = "cerebellum" 

dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
parc     = Parcellate()
simplt   = SimMatrixPlot()
nba      = NetBasedAnalysis()
nettools = NetTools()
mesim    = MeSiM()
###############################################################################
FONTSIZE       = 16



def _read_pairs_from_file(path):
    """Return list of (subject, session) extracted from TSV/CSV file."""
    delimiter = '\t' if path.lower().endswith('.tsv') else ','
    try:
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            headers = reader.fieldnames or []
            col_sub, col_ses = None, None
            for header in headers:
                lower = header.lower()
                if col_sub is None and lower in ("participant_id", "subject", "subject_id", "sub", "id"):
                    col_sub = header
                if col_ses is None and lower in ("session", "session_id", "ses"):
                    col_ses = header
            if col_sub is None or col_ses is None:
                raise ValueError("Participants file must contain subject and session columns.")

            pairs = []
            for row in reader:
                sid = str(row.get(col_sub, "")).strip()
                ses = str(row.get(col_ses, "")).strip()
                if not sid or not ses:
                    continue
                pairs.append((sid, ses))
    except FileNotFoundError:
        raise

    return list(dict.fromkeys(pairs))


def _get_metabolite_list(b0_strength):
    """Return ordered list of metabolites expected for the specified B0."""
    if b0_strength == 3:
        return ["CrPCr", "GluGln", "GPCPCh", "NAANAAG", "Ins"]
    if b0_strength == 7:
        return ["NAA", "NAAG", "Ins", "GPCPCh", "Glu", "Gln", "CrPCr", "GABA", "GSH"]
    raise ValueError(f"Unsupported B0 strength: {b0_strength}")


def _resolve_b0_argument(b0_argument):
    """
    Interpret the --b0 argument which can be either a B0 strength (3 or 7)
    or a custom list of metabolite names.
    Returns a tuple of (b0_strength or None, metabolite_list).
    """
    if b0_argument is None:
        return 3, _get_metabolite_list(3)

    values = b0_argument if isinstance(b0_argument, (list, tuple)) else [b0_argument]
    if len(values) == 1 and isinstance(values[0], str) and "," in values[0]:
        # Support comma-separated list passed as a single CLI token.
        values = [item.strip() for item in values[0].split(",") if item.strip()]

    if len(values) == 1:
        raw_value = values[0]
        try:
            b0_strength = float(raw_value)
        except (TypeError, ValueError):
            # Single non-numeric value → treat as custom metabolite list
            return None, [str(raw_value)]
        if b0_strength in (3, 7):
            return b0_strength, _get_metabolite_list(b0_strength)
        raise ValueError("--b0 numeric value must be either 3 or 7.")

    # Multiple values → custom metabolite list
    return None, [str(item) for item in values]


def _discover_all_pairs(group):
    """Return all subject-session pairs available for a group."""
    dataset_root = join(dutils.BIDSDATAPATH, group)
    pairs = []

    default_list = join(dataset_root, "participants_allsessions.tsv")
    if exists(default_list):
        try:
            pairs = _read_pairs_from_file(default_list)
            pairs = [pair for pair in pairs if pair[1] != "V2BIS"]
        except Exception as exc:
            debug.warning(f"Failed reading {default_list}: {exc}")

    if pairs:
        return pairs

    if not isdir(dataset_root):
        return []

    for entry in sorted(os.listdir(dataset_root)):
        if not entry.startswith("sub-"):
            continue
        subj_dir = join(dataset_root, entry)
        if not isdir(subj_dir):
            continue
        subject_id = entry[4:]
        for session_entry in sorted(os.listdir(subj_dir)):
            if not session_entry.startswith("ses-"):
                continue
            session_id = session_entry[4:]
            pairs.append((subject_id, session_id))

    return list(dict.fromkeys(pairs))


def _run_single_subject(args, subject_id, session):
    subject_id = str(subject_id)
    session = str(session)
    GROUP = args.group
    npert = args.npert
    OVERWRITE = args.overwrite
    LEAVE_ONE_OUT = bool(args.leave_one_out)
    NPROC = args.nthreads
    SHOW_PLOT = bool(args.show_plot)
    preproc_string = args.preproc
    scale = args.scale
    parc_scheme = args.parc
    t1mask_path_arg = args.t1mask
    outDirfigure_path = args.results_dir_path
    analyze_bool = bool(args.analyze)
    metabolites_list = getattr(args, "metabolites", METABOLITES)

    debug.info(f"GM Parcellation: {parc_scheme}")
    debug.info(f"Group: {GROUP}")
    debug.info(f"Number of perturbations: {npert}")
    debug.info(f"subject id: {subject_id}")
    debug.info(f"session: {session}")

    ftools = FileTools()

    sel_parcel_list = [
        "ctx-rh", "subc-rh", "thal-rh", "cer-rh",
        "ctx-lh", "subc-lh", "thal-lh", "cer-lh",
    ]

    prefix = f"sub-{subject_id}_ses-{session}"
    mridata = MRIData(subject_id, session, group=GROUP, metabolites=metabolites_list)
    outfilepath = mridata.get_connectivity_path(
        "mrsi", parc_scheme, scale, npert, filtoption=preproc_string.replace("filt", "")
    )
    connectome_dir_path = split(outfilepath)[0]

    if exists(outfilepath):
        debug.success(prefix, "Already processed")
        if OVERWRITE:
            debug.warning(prefix, "Overwriting existing")
        else:
            return

    mrsi_ref_img_path = mridata.get_mri_filepath(
        modality="mrsi", space="orig", desc="signal", met="Ins", option=preproc_string
    )
    if not exists(mrsi_ref_img_path):
        debug.error("No MRSI data found")
        return
   

    try:
        if exists(t1mask_path_arg):
            t1mask_path = t1mask_path_arg
        else:
            t1mask_path = mridata.find_nifti_paths(t1mask_path_arg)
            if t1mask_path is None:
                debug.error(f"{t1mask_path} does not exists or no matching pattern for {t1mask_path_arg} found")
                return 
    except Exception as e: 
        debug.error("--t1 argument must be a valid string or path")
        return


    debug.title(f"Compute Metabolic Simmilarity {prefix}")

    mrsi_orig_mask_nifti = mridata.get_mri_nifti(modality="mrsi", space="orig", desc="brainmask")
    mrsi_orig_mask_np = mrsi_orig_mask_nifti.get_fdata().squeeze()
    metadata = mridata.extract_metadata(t1mask_path)
    growmm   = args.grow
    
    
    parcel_mrsi_path = mridata.get_parcel_path(space="mrsi",
                                                            parc_scheme=parc_scheme,
                                                            scale=scale,
                                                            acq=metadata["acq"],
                                                            run=metadata["run"],
                                                            grow=growmm)

    if not exists(parcel_mrsi_path):
        debug.warning("Parcel image in",parc_scheme,scale,metadata["acq"],metadata["run"],
                      "does not exists, recreating one from atlas")
        atlas_str = f"chimera-{parc_scheme}-{scale}" if "cubic" not in parc_scheme else f"cubic-{scale}mm"
        atlas_mni_path = join(dutils.DEVANALYSEPATH,"data","atlas",
                                        f"{atlas_str}",f"{atlas_str}.nii.gz")
        # MNI -> ANAT
        transform_list   = mridata.get_transform("inverse","anat")
        parcel_t1_ants   = reg.transform(t1mask_path,atlas_mni_path,transform_list,
                                        interpolator_mode="genericLabel")
        # ANAT -> MRSI
        transform_list   = mridata.get_transform("inverse","mrsi")
        parcel_mrsi_np = reg.transform(mrsi_orig_mask_nifti,parcel_t1_ants,transform_list,
                                        interpolator_mode="genericLabel").numpy()
        parcel_mrsi_ni   = ftools.numpy_to_nifti(parcel_mrsi_np,header=mrsi_orig_mask_nifti.header)
        parcel_mrsi_path = mridata.get_parcel_path("mrsi",parc_scheme,scale,acq=metadata["acq"],run=metadata["run"],mode="atlas")
        debug.info("Saving MRSI space parcellation to",parcel_mrsi_path)
        ftools.save_nii_file(parcel_mrsi_ni,parcel_mrsi_path)
        # TSV
        tsv_source = atlas_mni_path.replace(".nii.gz",".tsv")
        tsv_dest   = parcel_mrsi_path.replace(".nii.gz",".tsv")
        shutil.copy(tsv_source, tsv_dest)         # copies file with metadata (permissions preserved)



    ########################################################################
    ##################### Discard low coverage parcels #####################
    ########################################################################
    debug.proc("Discard low coverage parcels")
    parcel_mrsi_ni = nib.load(parcel_mrsi_path)
    parcel_mrsi_np = parcel_mrsi_ni.get_fdata()
    parcel_mrsi_header = parcel_mrsi_ni.header
    parcel_header_dict = parc.get_parcel_header(parcel_mrsi_path.replace(".nii.gz", ".tsv"))
    transform_list = mridata.get_transform("inverse", "mrsi")
    t1mask_mrsi_img = reg.transform(mrsi_ref_img_path, t1mask_path, transform_list).numpy()
    parcel_header_dict = parc.count_voxels_per_parcel(
        parcel_mrsi_np, mrsi_orig_mask_np, t1mask_mrsi_img, parcel_header_dict
    )
    all_labels_list = [sub_dict['label'] for sub_dict in parcel_header_dict.values()]
    voxels_outside_mrsi = {k: v for k, v in parcel_header_dict.items() if v['count'][-1] <= 5}
    parcel_labels_ignore = [sub_dict['label'] for sub_dict in voxels_outside_mrsi.values()]
    parcel_label_ids_ignore = [keys for keys in voxels_outside_mrsi.keys()]
    label_list_concat = ["-".join(sublist) for sublist in all_labels_list]
    parcel_labels_ignore_concat = ["-".join(sublist) for sublist in parcel_labels_ignore]
    os.makedirs(connectome_dir_path, exist_ok=True)
    ########################################################################
    ######################### Metabolic Similarity #########################
    ########################################################################
    debug.proc("Metabolic Similarity")
    mrsirand = Randomize(mridata, space="orig", option=preproc_string)
    simmatrix_kwargs = dict(
        parcel_mrsi_np=parcel_mrsi_np,
        parcel_header_dict=parcel_header_dict,
        parcel_label_ids_ignore=parcel_label_ids_ignore,
        N_PERT=npert,
        corr_mode="spearman",
        rescale="zscore",
        n_proc=NPROC,
        leave_one_out=LEAVE_ONE_OUT,
    )
    (
        simmatrix_sp,
        pvalue_sp,
        parcel_concentrations,
        simmatrix_sp_leave_out,
    ) = mesim.compute_simmatrix_with_leaveout(mrsirand, **simmatrix_kwargs)

    del parcel_header_dict[0]
    labels_indices = np.array(list(parcel_header_dict.keys()))
    np.trim_zeros(labels_indices)
    simmatrix_ids_to_delete = []
    parcel_ids_positions, label_list_concat = parc.get_main_parcel_plot_positions(
        sel_parcel_list, label_list_concat
    )

    for idx_to_del in parcel_label_ids_ignore:
        simmatrix_ids_to_delete.append(np.where(labels_indices == idx_to_del)[0][0])

    os.makedirs(connectome_dir_path, exist_ok=True)
    np.savez(
        f"{outfilepath}",
        parcel_concentrations=parcel_concentrations,
        simmatrix_sp=simmatrix_sp,
        pvalue_sp=pvalue_sp,
        simmatrix_sp_leave_out=simmatrix_sp_leave_out,
        labels=label_list_concat,
        labels_indices=labels_indices,
        parcel_labels_ignore=parcel_labels_ignore_concat,
        simmatrix_ids_to_delete=simmatrix_ids_to_delete,
        metabolites_leaveout=metabolites_list,
    )

    ftools.save_dict(parcel_header_dict, outfilepath.replace(".npz", ".json"))
    debug.success(f"Results Saved to {outfilepath}")
    debug.separator()

    if args.msmode:
        # Dim-Reduction and project to MNI
        debug.proc("Metabolic Similarity Mode")
        features_1D  = nettools.dimreduce_matrix(simmatrix_sp,method='pca_tsne',output_dim=1,
                                                scale_factor=255.0)

        parcellation_img     = nib.load(parcel_mrsi_path.replace("space-mrsi","space-mni152"))
        header_mni152        = parcellation_img.header
        debug.proc("Projection to MNI152")
        label_indices_gm =  labels_indices[labels_indices<3000]
        projected_data_3D = nettools.project_to_3dspace(features_1D,
                                                    parcellation_img.get_fdata().astype(int),
                                                    label_indices_gm)
        outfilepath  = outfilepath.replace("desc-connectivity_mrsi.npz","desc-3Dmetabsim_mrsi.nii.gz")
        outpath      = outfilepath.replace("connectivty","msmode")
        ftools.save_nii_file(projected_data_3D,outpath=outpath,header=header_mni152)
        debug.info("MS mode NIFTI saved to ",outpath)

    if not analyze_bool:
        return

    parcel_mrsi = ftools.numpy_to_nifti(parcel_mrsi_np.astype(int), parcel_mrsi_header)
    centroids_dict = nettools.compute_parcel_centers(parcel_mrsi)
    centroids_arr = [
        centroids_dict[parcel_id]
        for parcel_id in parcel_header_dict.keys()
        if parcel_id not in parcel_label_ids_ignore
    ]
    distance_matrix = nettools.compute_distance_matrix(np.array(centroids_arr))
    try:
        results_filename = f"{prefix}_MeSiM_atlas-{parc_scheme}_scale-{scale}-npert-{npert}_richclub.pdf"
        if outDirfigure_path is None:
            outDirfigure_path = join(dutils.ANARESULTSPATH, "MeSiMs", GROUP, prefix)
            debug.info("Setting result figures output directory to ", outDirfigure_path)
        alpha = 0.05 / npert
        simmatrix_sp_corr = copy.deepcopy(simmatrix_sp)
        simmatrix_sp_corr[pvalue_sp > alpha] = 0
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))
        os.makedirs(outDirfigure_path, exist_ok=True)
        simplt.plot_simmatrix(
            simmatrix_sp,
            ax=axs[0, 0],
            titles=f"MeSiM",
            scale_factor=0.4,
            parcel_ids_positions=parcel_ids_positions,
            colormap="bluewhitered",
            show_parcels="H",
        )
        simplt.plot_simmatrix(
            simmatrix_sp_corr,
            ax=axs[0, 1],
            titles=f"MeSiM corrected p>{alpha}",
            scale_factor=0.4,
            parcel_ids_positions=parcel_ids_positions,
            colormap="bluewhitered",
            show_parcels="H",
        )

        debug.info("Create Rich-Club Results")
        simmatrix_adjusted = copy.deepcopy(simmatrix_sp)
        array_after_row_deletion = np.delete(simmatrix_adjusted, simmatrix_ids_to_delete, axis=0)
        simmatrix_adjusted = np.delete(array_after_row_deletion, simmatrix_ids_to_delete, axis=1)
        non_zero_indices = np.where(simmatrix_adjusted.sum(axis=0) != 0)[0]
        simmatrix_adjusted = simmatrix_adjusted[non_zero_indices[:, None], non_zero_indices]
        density = 0.18
        simmatrix_binarized = nba.binarize(
            simmatrix_adjusted, threshold=density, mode="abs", threshold_mode="density"
        )
        iu = np.triu_indices_from(simmatrix_adjusted, k=1)
        simmatrix_upper = simmatrix_adjusted[iu]
        distance_upper = distance_matrix[iu]
        mask_pos = simmatrix_upper > 0
        mask_neg = simmatrix_upper < 0
        df_pos = pd.DataFrame(
            {
                "Distance": distance_upper[mask_pos].flatten(),
                "Correlation": simmatrix_upper[mask_pos].flatten(),
                "Type": "Positive",
            }
        )
        df_neg = pd.DataFrame(
            {
                "Distance": distance_upper[mask_neg].flatten(),
                "Correlation": simmatrix_upper[mask_neg].flatten(),
                "Type": "Negative",
            }
        )
        df = pd.concat([df_pos, df_neg])

        RC_ALPHA = 0.05
        np.fill_diagonal(simmatrix_binarized, 0)
        node_deg, rc_coef, rand_rc_params = nba.compute_richclub_stats(
            simmatrix_binarized, alpha=RC_ALPHA
        )
        median_rand_rc = rand_rc_params["median"]
        lower_rand_rc = rand_rc_params["lower"]
        upper_rand_rc = rand_rc_params["upper"]

        simplt.plot_simmatrix(
            simmatrix_adjusted,
            ax=axs[1, 0],
            titles=f"MeSiM [removed empty nodes]",
            scale_factor=0.4,
            parcel_ids_positions=parcel_ids_positions,
            colormap="blueblackred",
        )
        simmatrix_binarized_pn = nba.binarize(
            simmatrix_adjusted, threshold=density, mode="posneg", threshold_mode="density"
        )
        simplt.plot_simmatrix(
            simmatrix_binarized_pn,
            ax=axs[1, 1],
            titles=f"Binarized MeSiM [density={density}]",
            parcel_ids_positions=parcel_ids_positions,
            colormap="blueblackred",
            scale_factor=0.6,
        )
        sns.kdeplot(
            data=df[df["Type"] == "Negative"],
            x="Distance",
            y="Correlation",
            fill=True,
            cmap="Blues",
            alpha=1,
            cbar=True,
            cbar_kws={"label": "Negative MeSiM KDE"},
            ax=axs[2, 0],
        )
        sns.kdeplot(
            data=df[df["Type"] == "Positive"],
            x="Distance",
            y="Correlation",
            fill=True,
            cmap="Reds",
            alpha=1,
            cbar=True,
            cbar_kws={"label": "Positive MeSiM KDE"},
            ax=axs[2, 0],
        )
        axs[2, 0].grid()
        axs[2, 0].set_xlabel('Distance', fontsize=FONTSIZE)
        axs[2, 0].set_ylabel('Correlations', fontsize=FONTSIZE)
        axs[2, 1].fill_between(
            node_deg,
            lower_rand_rc,
            upper_rand_rc,
            color='gray',
            alpha=0.5,
            label=f"Null Distribution (α={RC_ALPHA})",
        )
        axs[2, 1].plot(node_deg, median_rand_rc, '--', color='black')
        axs[2, 1].plot(node_deg, rc_coef, label='Metabolic Network', color='green')
        axs[2, 1].set_xlabel('Degree', fontsize=FONTSIZE)
        axs[2, 1].set_ylabel('Rich-Club Coefficient', fontsize=FONTSIZE)
        axs[2, 1].legend()
        axs[2, 1].grid()
        plot_outpath = join(outDirfigure_path, results_filename)
        fig.savefig(plot_outpath)
        plt.tight_layout()
        if SHOW_PLOT:
            plt.show()
    except Exception as e:
        debug.error("Failed creating results", e)
        return
    



def main():
    global METABOLITES
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    parser.add_argument('--parc', type=str, default="LFMIHIFIS", choices=['LFMIHIFIS', 'LFMIHIFIF', 'LFIIIIFIS','LFMIHISIFF'],
                        help='Chimera parcellation scheme, choice must be one of: LFMIHIFIS [default], LFMIHIFIF')
    parser.add_argument('--scale', type=int, default=3, help="Cortical parcellation scale (default: 3)")
    parser.add_argument('--grow',type=int,default=2,help="Gyral WM grow into GM in mm (default: 2)")
    parser.add_argument('--nthreads', type=int, default=4, help="Number of parallel threads (default: 4)")
    parser.add_argument('--npert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument(
        '--b0',
        nargs='+',
        default=["3"],
        help=(
            "Either specify 3 or 7 to auto-select metabolites for that B0 field "
            "strength, or provide a custom list of metabolite names (comma-separated or space-separated)."
        ),
    )

    parser.add_argument('--group', type=str, default="Mindfulness-Project")
    parser.add_argument('--subject_id', type=str, help='subject id', default="S002")
    parser.add_argument('--session', type=str, help='recording session', default="V3")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--leave_one_out', action='store_true', help="Leave-one-metaobolite-out (default: 0)")
    parser.add_argument('--show_plot',action='store_true', help="Display similarity matrix plot (default: 0)")
    parser.add_argument('--preproc', type=str, default="filtbiharmonic_pvcorr_GM", help="Preprocessing of orig MRSI files (default: filtbiharmonic)")
    parser.add_argument('--t1mask', type=str, default="desc-brainmask_T1w", help="Anatomical T1w brain mask path or pattern")
    parser.add_argument('--results_dir_path', type=str, default=None,
                        help=f"Directory path where results figures will be saved (default: {dutils.ANARESULTSPATH})")
    parser.add_argument('--analyze', action='store_true', help="Apply RichClub Analysis (default: 0)")
    parser.add_argument('--msmode', action='store_true', help="Compute 1st Metabolic Similarity Mode")
    parser.add_argument('--participants', type=str, default=None,
                        help="Path to TSV/CSV containing subject-session pairs to process in batch.")
    parser.add_argument('--batch', type=str, default='off', choices=['off', 'all', 'file'],
                        help="Batch mode: 'all' uses all available subject-session pairs; 'file' uses --participants; 'off' processes a single couplet.")

    args = parser.parse_args()

    try:
        b0_strength, metabolites = _resolve_b0_argument(args.b0)
    except ValueError as exc:
        debug.error(str(exc))
        return
    METABOLITES = metabolites
    args.b0_strength = b0_strength
    args.metabolites = metabolites

    if args.batch == 'off':
        pair_list = [(args.subject_id, args.session)]
    elif args.batch == 'file':
        if not args.participants:
            debug.error("--batch=file requires --participants to be provided.")
            return
        try:
            pair_list = _read_pairs_from_file(args.participants)
        except Exception as exc:
            debug.error(f"Failed to read participants list: {exc}")
            return
    else:
        pair_list = _discover_all_pairs(args.group)
        if not pair_list:
            debug.error(f"No subject-session pairs found for group {args.group}.")
            return

    if not pair_list:
        debug.error("No subject-session pairs to process.")
        return

    total = len(pair_list)
    for index, (subject_id, session) in enumerate(pair_list, start=1):
        run_args = argparse.Namespace(**vars(args))
        run_args.subject_id = subject_id
        run_args.session = session
        if args.batch != 'off':
            debug.title(f"Batch item {index}/{total}: sub-{subject_id}_ses-{session}")
        try:
            _run_single_subject(run_args, subject_id, session)
        except Exception as exc:
            import traceback
            debug.error(traceback.format_exc())
            debug.error(f"Processing sub-{subject_id}_ses-{session} failed: {exc}")


if __name__ == "__main__":
    main()





    
