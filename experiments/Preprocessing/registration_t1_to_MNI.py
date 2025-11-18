import os, sys, csv
import numpy as np
from registration.registration import Registration
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, exists, isdir
from tools.mridata import MRIData
import argparse
import nibabel as nib
from nilearn import datasets  



dutils       = DataUtils()
debug        = Debug()
reg          = Registration()


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


def _run_single_registration(args, subject_id, session):
    group = args.group
    nthreads = args.nthreads
    t1_path_arg = args.t1
    overwrite_flag = bool(args.overwrite)

    subject_id = str(subject_id)
    session = str(session)

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

    mridata = MRIData(subject_id, session, group)
    bids_root_path = join(dutils.BIDSDATAPATH, group)
    ants_transform_path = join(bids_root_path, "derivatives", "transforms", "ants")

    debug.separator()
    debug.proc(f"Processing {subject_id}-{session} ")

    try:
        if exists(t1_path_arg):
            t1_path = t1_path_arg
        else:
            t1_path = mridata.find_nifti_paths(t1_path_arg)
            if t1_path is None:
                debug.error(f"{t1_path} does not exists or no matching pattern for {t1_path_arg} found")
                return
            else:
                debug.info("Using t1 image",t1_path) 
    except Exception as e: 
        debug.error("--t1 argument must be a valid string or path")
        return

    t1_resolution = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
    debug.info("Loading MNI with", t1_resolution, "mm isotropic resolution")
    mni_template = datasets.load_mni152_template(t1_resolution)

    transform_dir_path = join(ants_transform_path, f"sub-{subject_id}", f"ses-{session}", "anat")
    transform_prefix = f"sub-{subject_id}_ses-{session}_desc-t1w_to_mni"
    transform_dir_prefix_path = join(transform_dir_path, transform_prefix)

    if not exists(f"{transform_dir_prefix_path}.syn.nii.gz") or overwrite_flag:
        debug.warning(f"{transform_prefix} not found or not up to date")
        syn_tx, _ = reg.register(
            fixed_input=mni_template,
            moving_input=t1_path,
            fixed_mask=None,
            moving_mask=None,
            transform="s",
            verbose=0
        )
        os.makedirs(transform_dir_path, exist_ok=True)
        reg.save_all_transforms(syn_tx, transform_dir_prefix_path)
    else:
        debug.success(f"{transform_prefix} to T1w Registration already done")
        return

    debug.title("DONE")
    debug.separator()


def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    parser.add_argument('--group', type=str, default="Mindfulness-Project")
    parser.add_argument('--nthreads', type=int, default=4, help="Number of CPU threads [default=4]")
    parser.add_argument('--subject_id', type=str, help='subject id', default="S002")
    parser.add_argument('--session', type=str, help='recording session', default="V3")
    parser.add_argument('--overwrite', type=int, default=0, choices=[1, 0], help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--t1', type=str, default=None, help="Anatomical T1w file path or patterns")
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











    

