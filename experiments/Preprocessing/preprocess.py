import os, sys, argparse, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from registration.registration import Registration
from tools.datautils import DataUtils
from os.path import split, join, exists, isdir
from tools.filetools import FileTools
from tools.debug import Debug
from tools.mridata import MRIData
from nilearn import datasets
import nibabel as nib
from rich.progress import Progress, TaskID
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from filters.pve import PVECorrection
from filters.biharmonic import BiHarmonic


dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
ftools   = FileTools()
pvc      = PVECorrection()
bhfilt   = BiHarmonic()


def _read_pairs_from_file(path):
    """Return list of (subject, session) extracted from TSV/CSV file."""
    delimiter = '\t' if path.lower().endswith('.tsv') else ','
    try:
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            headers = reader.fieldnames or []
            col_sub, col_ses = None, None
            for header in headers:
                h_lower = header.lower()
                if col_sub is None and h_lower in ("participant_id", "subject", "subject_id", "sub", "id"):
                    col_sub = header
                if col_ses is None and h_lower in ("session", "session_id", "ses"):
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

    # Preserve order while deduplicating
    unique_pairs = list(dict.fromkeys(pairs))
    return unique_pairs


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

    # Deduplicate while keeping order
    return list(dict.fromkeys(pairs))


def filter_worker(input_path, output_path, mask_path,percentile):
    try:
        image_og_nifti   = nib.load(input_path)
        brain_mask       = nib.load(mask_path)
        image_filt_nifti = bhfilt.proc(image_og_nifti,brain_mask,fwhm=None,percentile=percentile)
        ftools.save_nii_file(image_filt_nifti, outpath=output_path)
        return output_path  # success marker
    except Exception as e:
        return {"transform_worker error": str(e), "outpath": output_path}

def transform_worker(fixed_image, moving_image, transform_list, outpath):
    try:
        out_nifti = reg.transform(fixed_image, moving_image, transform_list).to_nibabel()
        os.makedirs(split(outpath)[0], exist_ok=True)
        ftools.save_nii_file(out_nifti, outpath=outpath)
        return outpath  # success marker
    except Exception as e:
        return {"transform_worker error": str(e), "outpath": outpath}

def pv_correction_worker(mridb,mrsi_nocorr_path,pv_corr_space="mrsi"):
    try:
        out_dict = pvc.proc(mridb, mrsi_nocorr_path, tissue_mask_space=pv_corr_space)
        return out_dict
    except Exception as e:
        return {"pv_correction_worker error": str(e), "outpath": mrsi_nocorr_path}




def _run_single_preprocess(args, subject_id, session):
    GROUP              = args.group
    filtoption         = args.filtoption
    overwrite          = bool(args.overwrite)
    overwrite_filt     = bool(args.overwrite_filt)
    overwrite_pvcorr   = bool(args.overwrite_pve)
    t1_path_arg        = args.t1
    B0_strength        = args.b0
    nthreads            = args.nthreads
    spike_pc           = args.spikepc
    pv_corr_str        = "pvcorr"
    verbose            = bool(args.v)
    TISSUE_LIST        = [None, "GM","WM","CSF"]


    if verbose:
        arg_dict = vars(args).copy()
        arg_dict.update({"subject_id": subject_id, "session": session})
        debug.display_dict(arg_dict,"MRSI Preprocessing Pipeline")

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
    subject_id = str(subject_id)
    session = str(session)
    recording_id = f"sub-{subject_id}_ses-{session}"

    if B0_strength == 3:
        METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]
    elif B0_strength == 7:
        METABOLITE_LIST = [
            "NAA", "NAAG", "Ins", "GPCPCh", "Glu", "Gln", "CrPCr", "GABA", "GSH"]


    # Use a list comprehension to generate two entries per metabolite:
    SIGNAL_LIST = [
        [met, "signal", filtoption] if entry == "signal" else [met, "crlb", None]
        for met in METABOLITE_LIST
        for entry in ("signal", "crlb")
    ] + [
        ["water", "signal", None],
        [None, "snr", None],
        [None, "fwhm", None],
        [None, "brainmask", None]
    ]

    
    mridata = MRIData(subject_id, session,GROUP)
    # Check if already processed
    try:
        if exists(t1_path_arg):
            t1_path = t1_path_arg
        else:
            t1_path = mridata.find_nifti_paths(t1_path_arg)
            if t1_path is None:
                debug.error(f"{t1_path} does not exists or no matching pattern for {t1_path_arg} found")
                return 
    except Exception as e: 
        debug.error("--t1 argument must be a valid string or path")
        return


    log_report = []
    ################################################################################
    #################### Filter MRSI spike ####################
    ################################################################################
    __path =  mridata.get_mri_filepath(modality="mrsi", space="orig", desc="signal", 
                                       met=METABOLITE_LIST[-1], option=filtoption)
    debug.info("Filter MRSI orig space spikes")
    if not exists(__path) or overwrite_filt:
        with Progress() as progress:
            task = progress.add_task("Filtering...", total=len(METABOLITE_LIST)+1)
            with ProcessPoolExecutor(max_workers=nthreads) as executor:
                futures = []
                for component in SIGNAL_LIST:
                    met, desc, option = component
                    if desc!="signal": continue
                    try:
                        # MRSI to T1W
                        mrsi_img_orig_path = mridata.get_mri_filepath(
                            modality="mrsi", space="orig", desc=desc, met=met,
                        )
                        mrsi_img_orig_filt_path = mridata.get_mri_filepath(
                            modality="mrsi", space="orig", desc=desc, met=met, option=filtoption
                        )
                        mask_path = mridata.get_mri_filepath(modality="mrsi",space="orig",desc="brainmask")
                        futures.append(executor.submit(
                            filter_worker,
                            mrsi_img_orig_path,
                            mrsi_img_orig_filt_path,
                            mask_path,
                            spike_pc,
                        ))
                        
                    except Exception as e:
                        debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                        progress.advance(task)

                # As each job completes, update the progress bar
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error(f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"Collecting results")
                    progress.advance(task)
    else:
        debug.success("Already transformed: SKIP")


    ################################################################################
    #################### Partial Volume Correction ####################
    ################################################################################
    debug.info("Partial Volume Correction")
    _e = True
    for i in [1,2,3]:
        try:
            _e&=exists(mridata.find_nifti_paths(f"_desc-p{i}_T1w"))
        except:
            _e&=False
    if _e: 
        __path = mridata.get_mri_filepath(modality="mrsi", space="orig", desc="signal", 
                                        met=METABOLITE_LIST[-1], option=f"{filtoption}_pvcorr")
        if not exists(__path) or overwrite_pvcorr:
            with Progress() as progress:
                task = progress.add_task("Partial volume correction...", total=len(METABOLITE_LIST))
                mridata = MRIData(subject_id, session,GROUP)
                with ProcessPoolExecutor(max_workers=nthreads) as executor:
                    futures = []
                    for component in SIGNAL_LIST:
                        met, desc, option = component
                        if desc!="signal" or option is None:continue
                        try:
                            if met=="water" and option is None: option=""
                            mrsi_nocorr_path = mridata.get_mri_filepath(
                                modality="mrsi", space="orig", desc=desc, met=met, option=option
                            )
                            if not exists(mrsi_nocorr_path):
                                log_report.append(f"Partial Volume Correction: no mrsi-space-orig found {component}")
                                continue
                            futures.append(executor.submit(
                                pv_correction_worker,
                                mridata,
                                mrsi_nocorr_path))
                        except Exception as e:
                            # debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                            log_report.append(f"Error preparing task: {recording_id} - {met, desc, option} Exception {e}")
                            progress.advance(task)
                    # As each job completes, update V   the progress bar
                    for future in as_completed(futures):
                        result = future.result()
                        if isinstance(result, dict) and "error" in result:
                            debug.error(f"PV Correction failed: {result['outpath']}", result["error"])
                        progress.update(task, description=f"Collecting results")
                        progress.advance(task)
        else:
            debug.success("Partial volume effect already corrected: SKIP")
    else:
        debug.error("One or multiple partial volume files p1 p2 p3 not found: Skip")
        log_report.append(f"Error preparing task: {recording_id} - {met, desc, option} Exception {e}")





    #########################################################################
    ########## MRSI-orig PV correction --> MRSI-T1W PV correction ###########
    #########################################################################
    __path =  mridata.get_mri_filepath(modality="mrsi",space="T1w",desc="signal",
                                       met=METABOLITE_LIST[-1],option=f"{filtoption}_pvcorr")
    if not exists(__path) or overwrite_pvcorr:
        debug.info("MRSI-orig PV correction --> MRSI-T1W PV correction")
        t1_resolution    = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
        mni_ref          = datasets.load_mni152_template(t1_resolution)
        transform_list   = mridata.get_transform("forward", "mrsi")

        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            futures = []
            for component in SIGNAL_LIST:
                met, desc, option = component
                if desc!="signal" or option is None:continue
                try:
                    for tissue in TISSUE_LIST:
                        preproc_str = f"{filtoption}_pvcorr_{tissue}" if tissue is not None else f"{filtoption}_pvcorr"
                        mrsi_orig_corr_path = mridata.get_mri_filepath(
                            modality="mrsi", space="orig", desc=desc, met=met, option=preproc_str
                        )
                        mrsi_img_corr_t1w_path = mridata.get_mri_filepath(
                            modality="mrsi", space="T1w", desc=desc, met=met, option=preproc_str
                        )
                        # debug.info(exists(mrsi_anat_corr_nifti),split(mrsi_anat_corr_nifti)[1])
                        if not exists(mrsi_orig_corr_path): 
                            # debug.error("\n","PV corrected MRSI orig-space does not exists")
                            # debug.error("\n",split(mrsi_orig_corr_path)[1],"not found ")
                            log_report.append(f"Transform MRSI-orig PV corrected --> MRSI-T1W PV corrected: no mrsi-space-orig-corr found {component} - {tissue}")
                            continue
                        if not exists(mrsi_img_corr_t1w_path) or overwrite_pvcorr: 
                            futures.append(executor.submit(
                                transform_worker,
                                t1_path,
                                mrsi_orig_corr_path,
                                transform_list,
                                mrsi_img_corr_t1w_path
                            ))
                        else: continue
                except Exception as e:
                    debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)

            # As each job completes, update the progress bar
            with Progress() as progress:
                task = progress.add_task("Correcting...", total=len(futures))
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error(f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"Collecting results")
                    progress.advance(task)
    else:
        debug.success("MRSI orig PV corrected already trasnformed to T1W space")



    ################################################################################
    #################### MRSI-orig --> MRSI-anat ####################
    ################################################################################
    __path =  mridata.get_mri_filepath(modality="mrsi",space="T1w",desc="signal",
                                       met=METABOLITE_LIST[-1],option=filtoption)
    debug.info("MRSI-orig --> MRSI-anat")
    if not exists(__path) or overwrite:
        transform_list   = mridata.get_transform("forward", "mrsi")
        with Progress() as progress:
            task = progress.add_task("Transforming...", total=len(SIGNAL_LIST))
            with ProcessPoolExecutor(max_workers=nthreads) as executor:
                futures = []
                for component in SIGNAL_LIST:
                    met, desc, option = component
                    try:
                        # MRSI to T1W
                        mrsi_img_orig_path = mridata.get_mri_filepath(
                            modality="mrsi", space="orig", desc=desc, met=met, option=option
                        )
                        mrsi_img_anat_path = mridata.get_mri_filepath(
                            modality="mrsi", space="T1w", desc=desc, met=met, option=option
                        )
                        if not exists(mrsi_img_orig_path):
                            log_report.append(f"Transform MRSI-orig --> MRSI-anat: no mrsi-space-{option}-orig found {component}")
                            continue
                        futures.append(executor.submit(
                            transform_worker,
                            t1_path,
                            mrsi_img_orig_path,
                            transform_list,
                            mrsi_img_anat_path
                        ))
                        
                    except Exception as e:
                        debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                        progress.advance(task)

                # As each job completes, update the progress bar
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error(f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"Collecting results")
                    progress.advance(task)
    else:
        debug.success("Already transformed: SKIP")



    ############################################################
    #################### MRSI-anat --> MRSI-MNI ####################
    ################################################################################
    debug.info("MRSI-anat --> MRSI-MNI")
    __path =  mridata.get_mri_filepath(modality="mrsi",space="mni",desc="signal",
                                       met=METABOLITE_LIST[-1],option=filtoption)
    if not exists(__path) or overwrite:
        t1_resolution    = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
        mni_ref          = datasets.load_mni152_template(t1_resolution)
        transform_list   = mridata.get_transform("forward", "anat")
        with Progress() as progress:
            task = progress.add_task("Transforming...", total=len(SIGNAL_LIST))
            with ProcessPoolExecutor(max_workers=nthreads) as executor:
                futures = []
                for component in SIGNAL_LIST:
                    met, desc, option = component
                    try:
                        mrsi_anat_path = mridata.get_mri_filepath(
                            modality="mrsi", space="T1w", desc=desc, met=met, option=option
                        )
                        mrsi_img_mni_path = mridata.get_mri_filepath(
                            modality="mrsi", space="mni", desc=desc, met=met, option=option
                        )
                        if not exists(mrsi_anat_path): 
                            log_report.append(f"MRSI-anat --> MRSI-MNI: no mrsi-space-t1w found {component}")
                            continue
                        futures.append(executor.submit(
                            transform_worker,
                            mni_ref,
                            mrsi_anat_path,
                            transform_list,
                            mrsi_img_mni_path
                        ))
                    except Exception as e:
                        debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                        progress.advance(task)

                # As each job completes, update the progress bar
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error(f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"Collecting results")
                    progress.advance(task)
    else:
        debug.success("Already transformed: SKIP")
        


    #########################################################################
    ########## MRSI-anat PV correction --> MRSI-MNI PV correction ###########
    #########################################################################
    __path =  mridata.get_mri_filepath(modality="mrsi",space="mni",desc="signal",
                                       met=METABOLITE_LIST[-1],option=f"{filtoption}_pvcorr")
    if not exists(__path) or overwrite_pvcorr:
        debug.info("Transform MRSI-anat PV corrected --> MRSI-MNI PV corrected")
        t1_resolution    = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
        mni_ref          = datasets.load_mni152_template(t1_resolution)
        transform_list   = mridata.get_transform("forward", "anat")

        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            futures = []
            for component in SIGNAL_LIST:
                met, desc, option = component
                if desc!="signal" or option is None:continue
                try:
                    for tissue in TISSUE_LIST:
                        preproc_str = f"{filtoption}_pvcorr_{tissue}" if tissue is not None else f"{filtoption}_pvcorr"
                        mrsi_anat_corr_path = mridata.get_mri_filepath(
                            modality="mrsi", space="T1w", desc=desc, met=met, option=preproc_str
                        )
                        mrsi_img_corr_mni_path = mridata.get_mri_filepath(
                            modality="mrsi", space="mni", desc=desc, met=met, option=preproc_str
                        )
                        # debug.info(exists(mrsi_anat_corr_nifti),split(mrsi_anat_corr_nifti)[1])
                        if not exists(mrsi_anat_corr_path): 
                            # debug.error("\n","PV corrected MRSI t1w-space does not exists")
                            log_report.append(f"Transform MRSI-anat PV corrected --> MRSI-MNI PV corrected: no mrsi-space-t1w-corr found {component}-{tissue}")
                            continue
                        if not exists(mrsi_img_corr_mni_path) or overwrite_pvcorr: 
                            futures.append(executor.submit(
                                transform_worker,
                                mni_ref,
                                mrsi_anat_corr_path,
                                transform_list,
                                mrsi_img_corr_mni_path
                            ))
                        else:
                            debug.info("mrsi_img_corr_mni_path already exists")
                            continue
                except Exception as e:
                    debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)

            # As each job completes, update the progress bar
            with Progress() as progress:
                task = progress.add_task("Correcting...", total=len(futures))
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error(f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"Collecting results")
                    progress.advance(task)
    else:
        debug.success("Partial Volume effect already corrected in MNI space")


    if len(log_report)==0:
        debug.success(f"Processed {recording_id} without errors")
    else:
        debug.info("---------------- LOG REPORT ----------------")
        for i in log_report:
            debug.error(i)
    debug.separator()
    return


def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    parser.add_argument('--group', type=str, default="Mindfulness-Project")
    parser.add_argument('--subject_id', type=str, default="S002", help="Subject ID [sub-??]")
    parser.add_argument('--session', type=str, default="V3", help="Session [ses-??]")
    parser.add_argument('--nthreads', type=int, default=4, help="Number of CPU threads [default=4]")
    parser.add_argument('--filtoption', type=str, default="filtbiharmonic", help="MRSI filter option  [default=filtbihamonic]")
    parser.add_argument('--spikepc', type=float, default=99, help="Percentile for MRSI signal spike detection  [default=98]")
    parser.add_argument('--t1', type=str, default=None, help="Anatomical T1w file path")
    parser.add_argument('--b0', type=float, default=3, choices=[3, 7], help="MRI B0 field strength in Tesla [default=3]")
    parser.add_argument('--overwrite', type=int, default=0, choices=[1, 0], help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--overwrite_filt', type=int, default=0, choices=[1, 0], help="Overwrite MRSI filtering output (default: 0)")
    parser.add_argument('--overwrite_pve', type=int, default=0, choices=[1, 0], help="Overwrite partial volume correction (default: 0)")
    parser.add_argument('--v', type=int, default=0, choices=[1, 0], help="Verbose")
    parser.add_argument('--participants', type=str, default=None,
                        help="Path to TSV/CSV containing subject-session pairs to process in batch.")
    parser.add_argument('--batch', type=str, default='off', choices=['off', 'all', 'file'],
                        help="Batch mode: 'all' uses all available subject-session pairs; 'file' uses --participants; 'off' processes a single couplet.")

    args = parser.parse_args()

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
    else:  # args.batch == 'all'
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
            _run_single_preprocess(run_args, subject_id, session)
        except Exception as exc:
            debug.error(f"Processing sub-{subject_id}_ses-{session} failed: {exc}")

if __name__=="__main__":
    main()














    
