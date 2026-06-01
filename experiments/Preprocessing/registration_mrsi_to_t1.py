import os, sys, csv
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
# try:
#     import tensorflow as tf
#     tf.config.set_visible_devices([], 'GPU')
# except:
#     pass
from mrsitoolbox.registration.registration import Registration
from mrsitoolbox.tools.datautils import DataUtils
from os.path import split, join, exists, isdir
from mrsitoolbox.tools.filetools import FileTools
from mrsitoolbox.tools.debug import Debug
import argparse
from mrsitoolbox.tools.mridata import MRIData
from mrsitoolbox.filters.biharmonic import BiHarmonic
import nibabel as nib
from nibabel.orientations import inv_ornt_aff 


dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
bhfilt   = BiHarmonic()
ftools   = FileTools()


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

def orientation_worker(input_path,overwrite_og=False,output_path=None,transform=np.array([[1.,  1.],[0., -1.],[2.,  1.]])):
    try:
        img     = nib.load(input_path)
    except Exception as e:
        debug.error("orient_to_target: Error loading input path \n ",e)
    try:
        img_np  = img.get_fdata().squeeze()
        aff     = img.affine
        aff_new = aff @ inv_ornt_aff(transform, img_np.shape[:3])
        # Copy header, set qform/sform
        hdr     = img.header.copy()
        hdr.set_sform(aff_new, code=1) ; hdr.set_qform(aff_new, code=1)
        out_img = nib.Nifti1Image(img_np, aff_new, header=hdr)
    except Exception as e:
        debug.error("orient_to_target: Error setting new image orientation \n ",e)
    try:
        # out_reorient_path = input_path.replace(".nii.gz","_reoriented.nii.gz")
        if overwrite_og:
            backup_path = input_path.replace(".nii.gz","_backup.nii.gz")
            ftools.save_nii_file(img, backup_path)
            ftools.save_nii_file(out_img, input_path)
        elif not overwrite_og and output_path:
            ftools.save_nii_file(out_img, output_path)
        else:
            debug.error("orient_to_target: Outpath needs to be specfied if overwrite set to False \n ",e)
    except:
        debug.error("orient_to_target: Error saving new image orientation \n ",e)


def _run_single_registration(args, subject_id, session):
    GROUP = args.group
    t1_path_arg = args.t1
    B0_strength = args.b0
    overwrite_flag = bool(args.overwrite)
    METABOLITE_REF = args.ref_met
    correct_orientation = args.corr_orient
    subject_id = str(subject_id)
    session = str(session)

    mridata = MRIData(subject_id, session, GROUP)
    bids_root_path = join(dutils.BIDSDATAPATH, GROUP)
    ants_transform_path = join(bids_root_path, "derivatives", "transforms", "ants")
    os.makedirs(ants_transform_path, exist_ok=True)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(args.nthreads)

    if B0_strength == 3:
        metabolite_list = ["CrPCr", "GluGln", "GPCPCh", "NAANAAG", "Ins"]
    elif B0_strength == 7:
        metabolite_list = [
            "NAA", "NAAG", "Ins", "GPCPCh", "Glu", "Gln", "CrPCr", "GABA", "GSH"
        ]
    else:
        metabolite_list = [METABOLITE_REF]

    mrsi_ref_path = mridata.get_mri_filepath(modality="mrsi", space="orig", desc="signal", met=METABOLITE_REF)
    if not exists(mrsi_ref_path):
        debug.error("MRSI path does not exists")
        return

    debug.separator()
    debug.proc(f"Start Registration {subject_id}-{session}")

    filt_ref_path = mridata.get_mri_filepath(
        modality="mrsi", space="orig", desc="signal", met=metabolite_list[-1], option="filtbiharmonic"
    )
    if not exists(filt_ref_path):
        mrsi_mask_path = mridata.get_mri_filepath(modality="mrsi", space="orig", desc="brainmask")
        if exists(mrsi_mask_path):
            nifti_mask = nib.load(mrsi_mask_path)
        else:
            mrsi_tmp_path = mridata.get_mri_filepath(modality="mrsi", space="orig", desc="signal", met=METABOLITE_REF)
            tmp_nifti = nib.load(mrsi_tmp_path)
            tmp_np = tmp_nifti.get_fdata()
            np_mask = np.zeros_like(tmp_np)
            np_mask[tmp_np > 0] = 1
            nifti_mask = ftools.numpy_to_nifti(np_mask, tmp_nifti.header)
            ftools.save_nii_file(np_mask, outpath=mrsi_mask_path, header=tmp_nifti.header)

        brain_mask = nifti_mask.get_fdata().squeeze().astype(bool)
        for metabolite in metabolite_list:
            debug.info("Denoise Orig", metabolite)
            try:
                image_og_nifti = mridata.get_mri_nifti(modality="mrsi", space="orig", desc="signal", met=metabolite)
                image_filt_nifti = bhfilt.proc(image_og_nifti, brain_mask, fwhm=6, percentile=98)
            except Exception as e :
                debug.error("bhfilt.proc",e)
            mrsi_filt_refout_path = mridata.get_mri_filepath(
                modality="mrsi", space="orig", desc="signal", met=metabolite, option="filtbiharmonic"
            )
            ftools.save_nii_file(image_filt_nifti, outpath=f"{mrsi_filt_refout_path}")
        debug.success("Done")

   
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


    transform_prefix = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w"
    transform_dir_path = join(ants_transform_path, f"sub-{subject_id}", f"ses-{session}", "mrsi")
    transform_dir_prefix_path = join(transform_dir_path, transform_prefix)
    warpfilename = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w.syn.nii.gz"

    if not exists(join(transform_dir_path, warpfilename)) or overwrite_flag:
        if exists(join(transform_dir_path, warpfilename)) and overwrite_flag:
            debug.warning(f"Overwriting existing {METABOLITE_REF} to T1w Registration")
        elif not exists(join(transform_dir_path, warpfilename)):
            debug.warning(f"Creating new MRSI {METABOLITE_REF} to T1w Registration")
        
        input_path = mridata.get_mri_filepath(
                modality="mrsi",
                space="orig",
                desc="signal",
                met=METABOLITE_REF,
                option="filtbiharmonic"
            )
        if correct_orientation:
            tmp_path = join("/tmp",split(input_path)[1])
            orientation_worker(input_path,output_path=tmp_path,overwrite_og=False)
            input_path = tmp_path
        syn_tx, _ = reg.register(
            fixed_input=t1_path,
            moving_input=input_path,
            transform="sr",
            verbose=0
        )
        reg.save_all_transforms(syn_tx, transform_dir_prefix_path)
    else:
        debug.success("Already registered")
        return

    debug.title("DONE")
    debug.separator()


def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    parser.add_argument('--group', type=str, default="Mindfulness-Project")
    parser.add_argument('--nthreads', type=int, default=4, help="Number of CPU threads [default=4]")
    parser.add_argument('--b0', type=float, default=3, choices=[3, 7], help="MRI B0 field strength in Tesla [default=3]")
    parser.add_argument('--ref_met', type=str, default="CrPCr", help="Reference metabolite to be coregistered with T1 [CrPCr]")
    parser.add_argument('--subject_id', type=str, help='subject id', default="S002")
    parser.add_argument('--session', type=str, help='recording session', default="V3")
    parser.add_argument('--overwrite', type=int, default=0, choices=[1, 0], help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--t1', type=str, default=None, help="Anatomical T1w file path")
    parser.add_argument('--participants', type=str, default=None,
                        help="Path to TSV/CSV containing subject-session pairs to process in batch.")
    parser.add_argument('--batch', type=str, default='off', choices=['off', 'all', 'file'],
                        help="Batch mode: 'all' uses all available subject-session pairs; 'file' uses --participants; 'off' processes a single couplet.")
    parser.add_argument('--corr_orient', type=int, default=0, help="Correct for oblique FOV orientation [default=0]")
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
            _run_single_registration(run_args, subject_id, session)
        except Exception as exc:
            debug.error(f"Processing sub-{subject_id}_ses-{session} failed: {exc}")

if __name__=="__main__":
    main()












    


