import os, sys, argparse, csv, subprocess, time, tempfile
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
from rich.table import Table
from rich import box
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from filters.pve import PVECorrection
from filters.biharmonic import BiHarmonic
from nibabel.orientations import inv_ornt_aff 

dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
ftools   = FileTools()
pvc      = PVECorrection()
bhfilt   = BiHarmonic()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_candidate_log_dirs = [join(PROJECT_ROOT, "log"), join(PROJECT_ROOT, "logs")]
for _log_dir in _candidate_log_dirs:
    if exists(_log_dir):
        LOG_DIR = _log_dir
        break
else:
    LOG_DIR = _candidate_log_dirs[0]
os.makedirs(LOG_DIR, exist_ok=True)


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


def orientation_worker(input_path,overwrite_og=False,output_path=None,transform=np.array([[1.,  1.],[0., -1.],[2.,  1.]])):
    backup_path = input_path.replace(".nii.gz","_backup.nii.gz")
    if os.path.exists(backup_path):
        input_path = backup_path
        # debug.info("Revert back to wrongly oriented backup",backup_path)
    else:
        pass
        # debug.info("Correct Orientation and store original as",split(backup_path)[1])
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
            ftools.save_nii_file(img, backup_path)
            ftools.save_nii_file(out_img, input_path)
        elif not overwrite_og and output_path:
            ftools.save_nii_file(out_img, output_path)
        else:
            debug.error("orient_to_target: Outpath needs to be specfied if overwrite set to False \n ",e)
    except:
        debug.error("orient_to_target: Error saving new image orientation \n ",e)




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


def _get_metabolite_list(b0_strength):
    """Return ordered list of metabolites expected for the specified B0."""
    if b0_strength == 3:
        return ["CrPCr", "GluGln", "GPCPCh", "NAANAAG", "Ins"]
    if b0_strength == 7:
        return ["NAA", "NAAG", "Ins", "GPCPCh", "Glu", "Gln", "CrPCr", "GABA", "GSH"]
    raise ValueError(f"Unsupported B0 strength: {b0_strength}")


def _build_signal_list(metabolites, filtoption):
    """Return the SIGNAL_LIST definition shared across the pipeline."""
    signal_list = []
    for met in metabolites:
        signal_list.append([met, "signal", filtoption])
        signal_list.append([met, "crlb", None])
    signal_list.extend([
        ["water", "signal", None],
        [None, "snr", None],
        [None, "fwhm", None],
        [None, "brainmask", None],
    ])
    return signal_list


def _warn_missing_forward_transforms(mridata, recording_id):
    """Log warnings for missing forward transforms that will be generated later."""
    checks = [
        ("mrsi", "MRSI→T1w (mrsi forward)"),
        ("anat", "T1w→MNI (anat forward)"),
        ("template-mni", "Template→MNI152 (template-mni forward)"),
        ("t1-template", "T1w→Template (t1-template forward)"),
    ]
    for stage_key, label in checks:
        stage_paths = mridata.get_transform("forward", stage_key) or []
        missing = [path for path in stage_paths if not exists(path)]
        if missing:
            missing_str = ", ".join(missing)
            debug.warning(
                f"{recording_id}: missing {label} transform(s): {missing_str}. "
                "They will be computed during runtime."
            )


def _run_multivisit_registration(mni_ref_img, group, subject_id):
    """Run the longitudinal registration script to regenerate key transforms."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "registration_multivisit.sh")
    if not exists(script_path):
        raise FileNotFoundError(f"registration_multivisit.sh not found at {script_path}")

    dataset_root = join(dutils.BIDSDATAPATH, group)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        nib.save(mni_ref_img, tmp_path)

        cmd = [
            "bash",
            script_path,
            "-i",
            dataset_root,
            "--mni",
            tmp_path,
            "--subject",
            subject_id,
        ]
        debug.info("\t", f"Ensuring longitudinal transforms for sub-{subject_id} via registration_multivisit.sh")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            message = stderr or stdout or "registration_multivisit.sh failed without output"
            raise RuntimeError(message)
    finally:
        if tmp_path and exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _resolve_t1_path(mridata, t1_argument):
    """Resolve --t1 argument into an on-disk file for the current subject/session."""
    if t1_argument is None:
        raise ValueError("--t1 argument must be provided for preprocessing.")

    t1_candidate = str(t1_argument).strip()
    if not t1_candidate:
        raise ValueError("--t1 argument must be provided for preprocessing.")

    if exists(t1_candidate):
        return os.path.abspath(t1_candidate)

    resolved = mridata.find_nifti_paths(t1_candidate)
    if resolved and exists(resolved):
        return resolved

    recording_id = f"sub-{mridata.subject_id}_ses-{mridata.session}"
    raise FileNotFoundError(
        f"No anatomical file matching '{t1_candidate}' for {recording_id}"
    )


def _gather_input_requirements(args, subject_id, session):
    """Collect presence/absence information for files required by preprocessing."""
    mridata = MRIData(subject_id, session, args.group)
    recording_id = f"sub-{subject_id}_ses-{session}"
    requirements = []

    try:
        t1_path = _resolve_t1_path(mridata, args.t1)
        t1_entry = {
            "label": "T1w reference (--t1)",
            "path": t1_path,
            "status": True,
        }
    except Exception as exc:
        t1_entry = {
            "label": "T1w reference (--t1)",
            "path": args.t1,
            "status": False,
            "message": str(exc),
        }
    requirements.append(t1_entry)

    metabolites = _get_metabolite_list(args.b0)
    signal_list = _build_signal_list(metabolites, args.filtoption)
    mrsi_entries = []
    for met, desc, _ in signal_list:
        label = f"MRSI {desc}" + (f" ({met})" if met else "")
        path = mridata.get_mri_filepath(modality="mrsi", space="orig", desc=desc, met=met)
        status = bool(path) and exists(path)
        is_brainmask = desc == "brainmask"
        entry = {
            "label": label,
            "path": path,
            "status": status,
            "autogen": is_brainmask,
            "met": met,
            "desc": desc,
        }
        mrsi_entries.append(entry)
        requirements.append(entry)

    pv_entries = []
    tissue_lookup = {1: "GM", 2: "WM", 3: "CSF"}
    for idx in (1, 2, 3):
        pattern = f"_desc-p{idx}_T1w"
        pv_path = None
        message = None
        try:
            pv_path = mridata.find_nifti_paths(pattern)
        except Exception as exc:
            message = str(exc)
        status = bool(pv_path) and exists(pv_path)
        pv_entry = {
            "label": f"T1w partial volume map p{idx}",
            "path": pv_path,
            "status": status,
            "tissue": tissue_lookup[idx],
        }
        if message:
            pv_entry["message"] = message
        pv_entries.append(pv_entry)
        requirements.append(pv_entry)

    transform_entries = {}
    for stage_key in ("mrsi", "anat", "template-mni", "t1-template"):
        stage_paths = mridata.get_transform("forward", stage_key) or []
        transform_entries[stage_key] = {
            "paths": stage_paths,
            "status": bool(stage_paths) and all(exists(path) for path in stage_paths),
        }

    missing_any = any(not req["status"] for req in requirements)
    missing_non_autogen = any(
        (not req["status"]) and not req.get("autogen") for req in requirements
    )
    return {
        "recording_id": recording_id,
        "requirements": requirements,
        "t1": t1_entry,
        "mrsi": mrsi_entries,
        "pv": pv_entries,
        "transforms": transform_entries,
        "missing": missing_any,
        "missing_non_autogen": missing_non_autogen,
    }


def _preflight_batch_inputs(args, pair_list):
    """Check availability of inputs for every batch item and prompt before running."""
    debug.separator()
    debug.title("Preanalysis: checking for required inputs")
    results = []
    with debug.console.status("[bold]Gathering requirements...[/bold]", spinner="dots") as status:
        for sub, ses in pair_list:
            status.update(f"[cyan]Checking sub-{sub}_ses-{ses}[/cyan]")
            results.append(_gather_input_requirements(args, sub, ses))

    CHECK_MARK = "[green]✔[/green]"
    CROSS_MARK = "[red]X[/red]"
    PROC_MARK = "[orange3]PROC[/orange3]"
    NA_MARK = "[grey58]N/A[/grey58]"
    transform_columns = [
        ("mrsi", "MRSI→T1"),
        ("anat", "T1→MNI"),
        ("t1-template", "T1→Template"),
        ("template-mni", "Template→MNI"),
    ]
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, title="Input availability summary")
    table.add_column("Recording", style="cyan", no_wrap=True)
    table.add_column("T1w ref", justify="center")
    table.add_column("MRSI files", justify="center")
    table.add_column("Brainmask", justify="center")
    table.add_column("Tissue files", justify="center")
    for _, label in transform_columns:
        table.add_column(label, justify="center")

    missing_count = 0
    total_missing_files = 0
    missing_recordings = []
    subject_session_counts = {}
    for subject_id, _ in pair_list:
        subject_session_counts[subject_id] = subject_session_counts.get(subject_id, 0) + 1
    log_lines = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_header = f"Preanalysis summary generated {timestamp}"
    log_lines.append(log_header)
    log_lines.append("=" * len(log_header))

    for result in results:
        t1_entry = result["t1"]
        t1_cell = CHECK_MARK if t1_entry["status"] else CROSS_MARK
        countable_entries = [entry for entry in result["mrsi"] if not entry.get("autogen")]
        total_expected = len(countable_entries)
        found_count = sum(1 for entry in countable_entries if entry["status"])
        missing_mrsi_entries = [entry for entry in countable_entries if not entry["status"]]
        mrsi_color = "green" if found_count == total_expected else "red"
        mrsi_cell = f"[{mrsi_color}]{found_count}/{total_expected}[/{mrsi_color}]"

        brainmask_entries = [entry for entry in result["mrsi"] if entry.get("autogen")]
        brainmask_status = bool(brainmask_entries and brainmask_entries[0]["status"])
        brainmask_cell = CHECK_MARK if brainmask_status else PROC_MARK

        pv_entries = result["pv"]
        cat12_segments = []
        cat12_states = []
        for tissue in ("GM", "WM", "CSF"):
            tissue_entry = next((entry for entry in pv_entries if entry.get("tissue") == tissue), None)
            tissue_status = bool(tissue_entry and tissue_entry["status"])
            color = "green" if tissue_status else "red"
            cat12_segments.append(f"[{color}]{tissue}[/{color}]")
            cat12_states.append(f"{tissue}:{'OK' if tissue_status else 'MISS'}")
        cat12_cell = " ".join(cat12_segments)

        row_cells = [
            result["recording_id"],
            t1_cell,
            mrsi_cell,
            brainmask_cell,
            cat12_cell,
        ]
        transform_text_parts = []
        recording_subject = result["recording_id"].split("_")[0].replace("sub-", "", 1)
        subject_sessions = subject_session_counts.get(recording_subject, 1)
        for stage_key, label in transform_columns:
            transform_info = result["transforms"].get(stage_key, {})
            is_ready = transform_info.get("status")
            if stage_key == "t1-template" and subject_sessions <= 1:
                row_cells.append(NA_MARK)
                state_text = "N/A"
            elif stage_key == "template-mni" and subject_sessions <= 1:
                row_cells.append(NA_MARK)
                state_text = "N/A"
            else:
                row_cells.append(CHECK_MARK if is_ready else PROC_MARK)
                state_text = "READY" if is_ready else "PROC"
            transform_text_parts.append(f"{label}={state_text}")

        table.add_row(*row_cells)

        missing_component_text = "none"
        if missing_mrsi_entries:
            missing_component_text = ", ".join(
                f"{(entry.get('met') or 'global')}:{entry.get('desc')}"
                for entry in missing_mrsi_entries
            )

        log_line = (
            f"{result['recording_id']}: "
            f"T1={'FOUND' if t1_entry['status'] else 'MISSING'}, "
            f"MRSI={found_count}/{total_expected}, "
            f"Brainmask={'FOUND' if brainmask_status else 'PROC'}, "
            f"CAT12={'/'.join(cat12_states)}, "
            + ", ".join(transform_text_parts)
            + f", MRSI-missing={missing_component_text}"
        )
        log_lines.append(log_line)

        non_autogen_missing = [
            req for req in result["requirements"]
            if (not req["status"]) and not req.get("autogen")
        ]
        if non_autogen_missing:
            missing_count += 1
            total_missing_files += len(non_autogen_missing)
            missing_recordings.append(result["recording_id"])

    debug.separator()
    debug.console.print(table)
    if missing_count:
        affected = ", ".join(missing_recordings)
        debug.error(
            f"Detected {total_missing_files} missing files across {missing_count}/{len(results)} batch items. "
            f"Skipping processing for: {affected}"
        )
    else:
        debug.success("All required inputs found for every batch item.")

    log_filename = f"preflight_{timestamp}.log"
    log_path = join(LOG_DIR, log_filename)
    try:
        with open(log_path, "w") as handle:
            handle.write("\n".join(log_lines) + "\n")
        debug.info(f"Preanalysis log saved to {log_path}")
    except Exception as exc:
        debug.warning(f"Unable to write preanalysis log to {log_path}: {exc}")

    if not sys.stdin.isatty():
        debug.warning("Non-interactive session detected; continuing without confirmation prompt.")
        return True

    try:
        response = input("Continue with preprocessing batch? [y/N]: ")
    except EOFError:
        response = ""
    if response.strip().lower() not in ("y", "yes"):
        debug.info("Stopping preprocessing per user request after preanalysis.")
        return False
    return True


def filter_worker(input_path, output_path, mask_path,percentile):
    try:
        image_og_nifti   = nib.load(input_path)
        brain_mask       = nib.load(mask_path)
        image_filt_nifti = bhfilt.proc(image_og_nifti,brain_mask,fwhm=None,percentile=percentile)
        ftools.save_nii_file(image_filt_nifti, outpath=output_path)
        return output_path  # success marker
    except Exception as e:
        return {"filter_worker error": str(e), "outpath": output_path}

def transform_worker(fixed_image, moving_image, transform_list, outpath):
    # debug.info(moving_image,"-->",outpath)
    try:
        out_nifti = reg.transform(fixed_image, moving_image, transform_list).to_nibabel()
        # debug.info(out_nifti)
        os.makedirs(split(outpath)[0], exist_ok=True)
        ftools.save_nii_file(out_nifti, outpath=outpath)
        return outpath  # success marker
    except Exception as e:
        return {"transform_worker error": str(e), "outpath": outpath}

def pv_correction_worker(mridb,mrsi_nocorr_path,pv_corr_space="mrsi"): # pv_corr_space="mrsi"
    try:
        out_dict = pvc.proc(mridb, mrsi_nocorr_path, tissue_mask_space=pv_corr_space)
        return out_dict
    except Exception as e:
        return {"pv_correction_worker error": str(e), "outpath": mrsi_nocorr_path}




def _run_single_preprocess(args, subject_id, session):
    GROUP               = args.group
    filtoption          = args.filtoption
    t1_path_arg         = args.t1
    B0_strength         = args.b0
    nthreads            = args.nthreads
    spike_pc            = args.spikepc
    pv_corr_str         = "pvcorr"
    verbose             = bool(args.v)
    TISSUE_LIST         = [None, "GM","WM","CSF"]
    correct_orientation = args.corr_orient


    if verbose:
        arg_dict = vars(args).copy()
        arg_dict.update({"subject_id": subject_id, "session": session})
        debug.display_dict(arg_dict,"MRSI Preprocessing Pipeline")

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
    subject_id = str(subject_id)
    session = str(session)
    recording_id = f"sub-{subject_id}_ses-{session}"

    METABOLITE_LIST = _get_metabolite_list(B0_strength)
    SIGNAL_LIST = _build_signal_list(METABOLITE_LIST, filtoption)

    mridata = MRIData(subject_id, session,GROUP)
    # Check if already processed
    try:
        t1_path = _resolve_t1_path(mridata, t1_path_arg)
    except Exception as exc:
        debug.error(str(exc))
        return

    _warn_missing_forward_transforms(mridata, recording_id)

    log_report = []
    ################################################################################
    #################### Filter MRSI spike ####################
    ################################################################################
    debug.proc("Filter MRSI orig space spikes")
    need_filter = bool(args.overwrite_filt)
    if not need_filter:
        for met, desc, option in SIGNAL_LIST:
            if desc != "signal":
                continue
            target = mridata.get_mri_filepath(
                modality="mrsi",
                space="orig",
                desc=desc,
                met=met,
                option=filtoption,
            )
            if not exists(target):
                need_filter = True
                break

    if need_filter:
        with Progress() as progress:
            mask_path = mridata.get_mri_filepath(modality="mrsi",space="orig",desc="brainmask")
            if correct_orientation:
                orientation_worker(mask_path,overwrite_og=True)
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
                        if correct_orientation:
                            orientation_worker(mrsi_img_orig_path,overwrite_og=True)
                        futures.append(executor.submit(
                            filter_worker,
                            mrsi_img_orig_path,
                            mrsi_img_orig_filt_path,
                            mask_path,
                            spike_pc,
                        ))
                    except Exception as e:
                        debug.error("\t",f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                        progress.advance(task)

                # As each job completes, update the progress bar
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error("\t",f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"\t Collecting results")
                    progress.advance(task)
    else:
        debug.success("\t","Already processed: SKIP")



    #########################################################################
    ########## REGISTRATION: MRSI-orig  --> MRSI-T1W  ###########
    #########################################################################
    # __path =  mridata.get_mri_filepath(modality="mrsi",space="T1w",desc="signal",
    #                                    met=METABOLITE_LIST[-1],option=f"{filtoption}_pvcorr")
    # if not exists(__path) or overwrite_pvcorr:
    debug.proc("REGISTRATION: MRSI-orig  --> MRSI-T1W")
    t1_resolution    = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
    mni_ref          = datasets.load_mni152_template(t1_resolution)
    transform_list   = mridata.get_transform("forward", "mrsi")
    if not all(exists(path) for path in transform_list) or args.overwrite_t1_reg:
        if args.overwrite_t1_reg:
            debug.warning("\t","Overwrite existing MRSI->T1w registration")
        else:
            debug.warning("\t","Missing MRSI->T1w transforms; launching registration_mrsi_to_t1.")
        registration_script = os.path.abspath(
            join(os.path.dirname(__file__), "registration_mrsi_to_t1.py")
        )
        cmd = [
            sys.executable,
            registration_script,
            "--group", GROUP,
            "--subject_id", subject_id,
            "--session", session,
            "--nthreads", str(nthreads),
            "--b0", str(B0_strength),
            "--batch", "off",
            "--corr_orient", str(int(correct_orientation)),
        ]
        if t1_path:
            cmd.extend(["--t1", str(t1_path)])
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            debug.error("\t","registration_mrsi_to_t1.py failed", exc)
            return
        except Exception as exc:
            debug.error("\t","Unable to start registration_mrsi_to_t1.py", exc)
            return
        transform_list = mridata.get_transform("forward", "mrsi")
        if not all(exists(path) for path in transform_list):
            debug.error("\t","registration_mrsi_to_t1.py did not produce the expected transforms")
            return
    else:
        debug.success("\t","Already computed: SKIP")


    ################################################################################
    #################### Partial Volume Correction ####################
    ################################################################################
    debug.proc("Partial Volume Correction (MRSI space)")
    def _has_tissue_map(idx):
        try:
            path = mridata.find_nifti_paths(f"_desc-p{idx}_T1w")
            return bool(path) and exists(path)
        except Exception:
            return False

    cat12_available = all(_has_tissue_map(idx) for idx in (1, 2, 3))
    if cat12_available: 
        needs_pvcorr = bool(args.overwrite_pve)
        if not needs_pvcorr:
            for met, desc, option in SIGNAL_LIST:
                if desc != "signal":
                    continue
                base_option = option or ""
                pvcorr_option = f"{base_option}_pvcorr" if base_option else "pvcorr"
                candidate = mridata.get_mri_filepath(
                    modality="mrsi",
                    space="orig",
                    desc=desc,
                    met=met,
                    option=pvcorr_option,
                )
                if not exists(candidate):
                    needs_pvcorr = True
                    break
        if needs_pvcorr:
            pvc.create_3tissue_pev(mridata,space="mrsi")
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
                            debug.error("\t",f"PV Correction failed: {result['outpath']}", result["error"])
                        progress.update(task, description=f"\t Collecting results")
                        progress.advance(task)
        else:
            debug.success("\t","Already processed: SKIP")
    else:
        debug.error("\t","One or multiple partial volume files p1/p2/p3 not found: Skip")
        log_report.append(
            "Partial Volume Correction skipped: missing one or more CAT12 tissue maps (p1-p3)"
        )








    #########################################################################
    ########## MRSI-orig PV correction --> MRSI-T1W PV correction ###########
    #########################################################################
    
    transform_list   = mridata.get_transform("forward", "mrsi")
    if args.mrsi_t1wspace and all(exists(transform_list)):
        debug.proc("TRANSFORM MRSI-orig PV correction --> MRSI-T1W")
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
                            log_report.append(f"no mrsi-space-orig-corr found {component} - {tissue}")
                            continue
                        if not exists(mrsi_img_corr_t1w_path) or args.overwrite_pve: 
                            futures.append(executor.submit(
                                transform_worker,
                                t1_path,
                                mrsi_orig_corr_path,
                                transform_list,
                                mrsi_img_corr_t1w_path
                            ))
                        else: continue
                except Exception as e:
                    debug.error("\t",f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
            # As each job completes, update the progress bar
            with Progress() as progress:
                task = progress.add_task("\t Correcting...", total=len(futures))
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error("\t",f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"\t Collecting results")
                    progress.advance(task)




    ################################################################################
    #################### MRSI-orig --> MRSI-T1w-anat ####################
    ################################################################################

    transform_list   = mridata.get_transform("forward", "mrsi")
    if all(exists(path) for path in transform_list) and args.tr_mrsi_t1:
        debug.proc("TRANSFORM MRSI-orig --> MRSI-anat")
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
                        debug.error("\t",f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                        progress.advance(task)

                # As each job completes, update the progress bar
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error("\t",f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"\t Collecting results")
                    progress.advance(task)
    elif args.tr_mrsi_t1 and not all(exists(path) for path in transform_list):
        debug.proc("TRANSFORM MRSI-orig --> MRSI-anat")
        debug.error("\t","No MRSI->T1w transform found: SKIP")
    else: pass



    ############################################################
    #################### MRSI-anat --> MRSI-MNI ####################
    ################################################################################
    debug.proc("REGISTRATION: Anatomical T1w --> MNI")
    t1_resolution    = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
    mni_ref          = datasets.load_mni152_template(t1_resolution)
    transform_list   = mridata.get_transform("forward", "anat")
    if not all(exists(path) for path in transform_list) or args.overwrite_mni_reg:
        if args.overwrite_mni_reg:
            debug.warning("\t","Overwriting existing transforms")
        debug.warning("\t","Missing T1w->MNI transforms; launching registration_t1_to_MNI.")
        registration_script = os.path.abspath(
            join(os.path.dirname(__file__), "registration_t1_to_MNI.py")
        )
        cmd = [
            sys.executable,
            registration_script,
            "--group", GROUP,
            "--subject_id", subject_id,
            "--session", session,
            "--nthreads", str(nthreads),
            "--batch", "off",
        ]
        if t1_path:
            cmd.extend(["--t1", str(t1_path)])
        if args.overwrite_mni_reg:
            cmd.extend(["--overwrite", "1"])
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            debug.error("\t","registration_t1_to_MNI.py failed", exc)
            return
        except Exception as exc:
            debug.error("\t","Unable to start registration_t1_to_MNI.py", exc)
            return
        transform_list = mridata.get_transform("forward", "anat")
        if not all(exists(path) for path in transform_list):
            debug.error("\t","registration_t1_to_MNI.py did not produce the expected transforms")
            return
    else:
        debug.success("\t","Registration already computed: SKIP")

    if args.mni_no_pvc:
        transform_mrsi_to_t1_list = mridata.get_transform("forward", "mrsi")
        transform_t1_to_mni_list  = mridata.get_transform("forward", "anat")
        transform_list            = transform_t1_to_mni_list + transform_mrsi_to_t1_list
        with Progress() as progress:
            task = progress.add_task("Transforming...", total=len(SIGNAL_LIST))
            with ProcessPoolExecutor(max_workers=nthreads) as executor:
                futures = []
                for component in SIGNAL_LIST:
                    met, desc, option = component
                    try:
                        mrsi_orig_path = mridata.get_mri_filepath(
                            modality="mrsi", space="orig", desc=desc, met=met, option=option
                        )
                        mrsi_img_mni_path = mridata.get_mri_filepath(
                            modality="mrsi", space="mni", desc=desc, met=met, option=option
                        )
                        if not exists(mrsi_orig_path): 
                            log_report.append(f"MRSI-anat --> MRSI-MNI: no mrsi-orig-space found {component}")
                            continue
                        futures.append(executor.submit(
                            transform_worker,
                            mni_ref,
                            mrsi_orig_path,
                            transform_list,
                            mrsi_img_mni_path
                        ))
                    except Exception as e:
                        debug.error("\t",f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                        progress.advance(task)

                # As each job completes, update the progress bar
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error("\t",f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"\t Collecting results")
                    progress.advance(task)

        


    #########################################################################
    ########## MRSI-orig PV correction --> MRSI-MNI PV correction ###########
    #########################################################################
    t1_resolution    = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
    mni_ref          = datasets.load_mni152_template(t1_resolution)
    __path =  mridata.get_mri_filepath(modality="mrsi",space="mni",desc="signal",res=t1_resolution,
                                       met=METABOLITE_LIST[-1],option=f"{filtoption}_pvcorr")

    if t1_resolution<1:
        t1_resolution_str = f"{int(t1_resolution * 10):02d}"
    else:
        t1_resolution_str = str(t1_resolution)
    
    if not exists(__path) or args.overwrite_pve:
        debug.proc(f"TRANSFORM: MRSI-orig PV corrected --> MRSI-MNI @ {t1_resolution_str}mm")   
        transform_mrsi_to_t1_list = mridata.get_transform("forward", "mrsi")
        transform_t1_to_mni_list  = mridata.get_transform("forward", "anat")
        transform_list            = transform_t1_to_mni_list + transform_mrsi_to_t1_list

        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            futures = []
            for component in SIGNAL_LIST:
                met, desc, option = component
                if desc!="signal" or option is None:continue
                try:
                    for tissue in TISSUE_LIST:
                        preproc_str = f"{filtoption}_pvcorr_{tissue}" if tissue is not None else f"{filtoption}_pvcorr"
                        mrsi_orig_corr_path = mridata.get_mri_filepath(
                            modality="mrsi", space="orig", desc=desc, 
                            met=met, option=preproc_str,
                        )
                        mrsi_img_corr_mni_path = mridata.get_mri_filepath(
                            modality="mrsi", space="mni", desc=desc, 
                            met=met, option=preproc_str,res=t1_resolution_str,
                        )
                        if not exists(mrsi_orig_corr_path): 
                            log_report.append(f"No mrsi-origspace-corr found {component}-{tissue}")
                            continue
                        if not exists(mrsi_img_corr_mni_path) or args.overwrite_pve: 
                            futures.append(executor.submit(
                                transform_worker,
                                mni_ref,
                                mrsi_orig_corr_path,
                                transform_list,
                                mrsi_img_corr_mni_path
                            ))
                        else:
                            continue
                except Exception as e:
                    debug.error("\t",f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)

            # As each job completes, update the progress bar
            with Progress() as progress:
                task = progress.add_task("\t Correcting...", total=len(futures))
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error("\t",f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"\t Collecting results")
                    progress.advance(task)
    else:
        debug.success("\t","Already processed")


    #########################################################################
    # MRSI-orig PV correction --> MRSI-MNI PV correction @ Orig Resolution ##
    #########################################################################
    __ogres_path =  mridata.get_mri_filepath(modality="mrsi",space="orig",desc="signal",
                                       met=METABOLITE_LIST[-1],option=f"{filtoption}_pvcorr")
    orig_resolution           = np.array(nib.load(__ogres_path).header.get_zooms()[:3]).mean()
    orig_resolution_int       = int(round(orig_resolution))
    __path =  mridata.get_mri_filepath(modality="mrsi",space="mni",desc="signal",res=orig_resolution_int,
                                       met=METABOLITE_LIST[-1],option=f"{filtoption}_pvcorr")
    
    if not exists(__path) or args.overwrite_mni:
        debug.proc(f"TRANSFORM MRSI-orig PV correction --> MRSI-MNI @ Orig {orig_resolution_int}mm")
        mni_ref                   = datasets.load_mni152_template(orig_resolution)
        transform_mrsi_to_t1_list = mridata.get_transform("forward", "mrsi")
        transform_t1_to_mni_list  = mridata.get_transform("forward", "anat")
        transform_list            = transform_t1_to_mni_list + transform_mrsi_to_t1_list
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
                        mrsi_img_corr_mni_origres_path = mridata.get_mri_filepath(
                            modality="mrsi", space="mni", desc=desc, met=met, option=preproc_str,
                            res = orig_resolution_int,
                        )
                        # debug.info(exists(mrsi_anat_corr_nifti),split(mrsi_anat_corr_nifti)[1])
                        if not exists(mrsi_orig_corr_path): 
                            # debug.error("\n","PV corrected MRSI t1w-space does not exists")
                            log_report.append(f"MRSI-orig PV correction --> MRSI-MNI PV correction @ {orig_resolution_int}mm: no mrsi-origspace-corr found {component}-{tissue}")
                            continue
                        if not exists(mrsi_img_corr_mni_origres_path) or args.overwrite_mni: 
                            futures.append(executor.submit(
                                transform_worker,
                                mni_ref,
                                mrsi_orig_corr_path,
                                transform_list,
                                mrsi_img_corr_mni_origres_path
                            ))
                        else:
                            debug.info("\t","mrsi_img_corr_mni_origres_path already exists")
                            continue
                except Exception as e:
                    debug.error("\t",f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)

            # As each job completes, update the progress bar
            with Progress() as progress:
                task = progress.add_task("\t Correcting...", total=len(futures))
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and "error" in result:
                        debug.error("\t",f"Transform failed: {result['outpath']}", result["error"])
                    progress.update(task, description=f"\t Collecting results")
                    progress.advance(task)
    else:
        debug.success("\t","Already Processed")



    #########################################################################
    # MRSI-orig PV correction --> MRSI-MNI PV correction - Longitudinal ##
    #########################################################################
    if args.proc_mnilong:
        debug.proc(f"TRANSFORM MRSI-orig PV correction --> MRSI-MNI152 Longitudinal @ Orig {orig_resolution_int}mm")
        __ogres_path =  mridata.get_mri_filepath(modality="mrsi",space="orig",desc="signal",
                                        met=METABOLITE_LIST[-1],option=f"{filtoption}_pvcorr")
        orig_resolution           = np.array(nib.load(__ogres_path).header.get_zooms()[:3]).mean()
        orig_resolution_int       = int(round(orig_resolution))
        __path =  mridata.get_mri_filepath(modality="mrsi",space="mni152long",desc="signal",
                                        res=orig_resolution_int, met=METABOLITE_LIST[-1],
                                        option=f"{filtoption}_pvcorr")
        
        if not exists(__path):
            mni_ref           = datasets.load_mni152_template(orig_resolution)
            transform_list = []
            transform_stages = [
                ("template-mni", "Template→MNI"),
                ("t1-template", "T1→Template"),
                ("mrsi", "MRSI→T1"),
            ]
            for stage_key, label in transform_stages:
                stage_paths = mridata.get_transform("forward", stage_key) or []
                missing = [path for path in stage_paths if not exists(path)]
                if missing and stage_key in {"template-mni", "t1-template"}:
                    try:
                        _run_multivisit_registration(mni_ref, args.group, args.subject_id)
                    except Exception as exc:
                        debug.error(
                            f"Failed to regenerate {label} transforms via registration_multivisit.sh: {exc}"
                        )
                        return
                    stage_paths = mridata.get_transform("forward", stage_key) or []
                    missing = [path for path in stage_paths if not exists(path)]
                if missing:
                    debug.error(
                        f"Missing transforms for {label}: {', '.join(missing)}"
                    )
                    return
                transform_list.extend(stage_paths)
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
                            mrsi_img_corr_mnilong_origres_path = mridata.get_mri_filepath(
                                modality="mrsi", space="mni152long", desc=desc, met=met, option=preproc_str,
                                res = orig_resolution_int,
                            )
                            if not exists(mrsi_orig_corr_path): 
                                debug.error("\n","PV corrected MRSI t1w-space does not exists")
                                log_report.append(
                                    f"TRANSFORM MRSI-orig PV correction --> MRSI-MNI152 Longitudinal @ Orig "
                                    f"; {orig_resolution_int}mm: no mrsi-origspace-corr found {component}-{tissue}"
                                )
                                continue
                            if not exists(mrsi_img_corr_mnilong_origres_path) or args.overwrite_mnilong: 
                                futures.append(executor.submit(
                                    transform_worker,
                                    mni_ref,
                                    mrsi_orig_corr_path,
                                    transform_list,
                                    mrsi_img_corr_mnilong_origres_path
                                ))
                            else:
                                debug.info("\t","mrsi_img_corr_mnilong_origres_path already exists")
                                continue
                    except Exception as e:
                        debug.error("\t",f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)

                # As each job completes, update the progress bar
                with Progress() as progress:
                    task = progress.add_task("\t Correcting...", total=len(futures))
                    for future in as_completed(futures):
                        result = future.result()
                        if isinstance(result, dict) and "error" in result:
                            debug.error("\t",f"Transform failed: {result['outpath']}", result["error"])
                        progress.update(task, description=f"\t Collecting results")
                        progress.advance(task)
        else:
            debug.success("\t","Already Processed")



    if len(log_report)==0:
        debug.success(f"Processed {recording_id} without errors")
    else:
        debug.info("---------------- LOG REPORT ----------------")
        for i in log_report:
            debug.error(i)
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
    parser.add_argument('--tr_mrsi_t1', type=int, default=0, choices=[1, 0], help="Generate intermdiairy T1w-space files (default: 0)")
    parser.add_argument('--overwrite_filt', type=int, default=0, choices=[1, 0], help="Overwrite MRSI filtering output (default: 0)")
    parser.add_argument('--overwrite_pve', type=int, default=0, choices=[1, 0], help="Overwrite partial volume correction (default: 0)")
    parser.add_argument('--overwrite_t1_reg', type=int, default=0, choices=[1, 0], help="Overwrite MRSI -> T1w registration (default: 0)")
    parser.add_argument('--overwrite_mni_reg', type=int, default=0, choices=[1, 0], help="Overwrite T1w -> MNI registration (default: 0)")
    parser.add_argument('--overwrite_mni', type=int, default=0, choices=[1, 0], help="Overwrite transform to MNI orig res (default: 0)")
    parser.add_argument('--overwrite_mnilong', type=int, default=0, choices=[1, 0], help="Overwrite transform to MNI-Longitudinal orig res (default: 0)")
    parser.add_argument('--proc_mnilong', type=int, default=0, choices=[1, 0], help="Process transform to MNI-Longitudinal orig res (default: 0)")
    parser.add_argument('--mni_no_pvc', type=int, default=0, choices=[1, 0], help="Get non-partial-volume-corrected MNI maps (default: 0)")
    parser.add_argument('--mrsi_t1wspace', type=int, default=0, choices=[1, 0], help="Get MRSI maps in T1w space (default: 0)")
    parser.add_argument('--v', type=int, default=0, choices=[1, 0], help="Verbose")
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
    else:  # args.batch == 'all'
        pair_list = _discover_all_pairs(args.group)
        if not pair_list:
            debug.error(f"No subject-session pairs found for group {args.group}.")
            return

    if not pair_list:
        debug.error("No subject-session pairs to process.")
        return

    if args.batch != 'off':
        if not _preflight_batch_inputs(args, pair_list):
            return

    total = len(pair_list)
    for index, (subject_id, session) in enumerate(pair_list, start=1):
        requirements = _gather_input_requirements(args, subject_id, session)
        if not requirements["t1"]["status"]:
            debug.warning(
                f"Skipping sub-{subject_id}_ses-{session}: missing T1 reference "
                f"({requirements['t1']['path'] or 'Not found'})"
            )
            continue
        mrsi_present = any(
            req["status"] and not req.get("autogen") for req in requirements["mrsi"]
        )
        if not mrsi_present:
            debug.warning(
                f"Skipping sub-{subject_id}_ses-{session}: no MRSI files detected"
            )
            continue
        debug.title(f"Processing sub-{subject_id}_ses-{session}")
        debug.separator()
        pair_start = time.time()
        run_args = argparse.Namespace(**vars(args))
        run_args.subject_id = subject_id
        run_args.session = session
        if args.batch != 'off':
            debug.info(f"Batch item {index}/{total}: sub-{subject_id}_ses-{session}")
        try:
            _run_single_preprocess(run_args, subject_id, session)
            duration = time.time() - pair_start
            minutes, seconds = divmod(int(duration), 60)
            debug.success(f"Completion time: {minutes:02d} min {seconds:02d} sec")
            debug.separator();debug.separator()
        except Exception as exc:
            debug.error("\t",f"Processing sub-{subject_id}_ses-{session} failed: {exc}")

if __name__=="__main__":
    debug.separator()
    main()














    
