#!/usr/bin/env python3

import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

_QT_RELAUNCH_ENV = "MRSI_BIDS_GUI_QT_REEXEC"


def _iter_python_candidates():
    seen = set()
    candidate_names = ("python3", "python")
    search_dirs = [Path(path).expanduser() for path in os.environ.get("PATH", "").split(os.pathsep) if path]
    current = Path(sys.executable).resolve()
    extra_candidates = [
        Path(sys.base_prefix) / "bin" / "python3",
        Path(sys.base_prefix) / "bin" / "python",
        Path.home() / "anaconda3" / "bin" / "python3",
        Path.home() / "anaconda3" / "bin" / "python",
        Path.home() / "miniconda3" / "bin" / "python3",
        Path.home() / "miniconda3" / "bin" / "python",
        Path("/usr/bin/python3"),
        Path("/bin/python3"),
    ]

    for candidate in [current] + extra_candidates + [search_dir / name for search_dir in search_dirs for name in candidate_names]:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if not resolved.exists() or not os.access(str(resolved), os.X_OK):
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        yield str(resolved)


def _qt_runtime_available(python_executable):
    probe = (
        "import importlib.util as u, sys\n"
        "qt_ok = any(u.find_spec(name) for name in ('PyQt6', 'PyQt5'))\n"
        "deps_ok = all(u.find_spec(name) for name in ('nibabel', 'rich'))\n"
        "sys.exit(0 if qt_ok and deps_ok else 1)\n"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _maybe_reexec_with_qt():
    if os.environ.get(_QT_RELAUNCH_ENV):
        return False

    current = str(Path(sys.executable).resolve())
    for candidate in _iter_python_candidates():
        if candidate == current:
            continue
        if not _qt_runtime_available(candidate):
            continue
        env = dict(os.environ)
        env[_QT_RELAUNCH_ENV] = candidate
        os.execve(candidate, [candidate, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    return False


try:
    from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QComboBox,
        QDialog,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
        QHeaderView,
        QCheckBox,
    )
    QT_LIB = 6
except ImportError:
    try:
        from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
        from PyQt5.QtGui import QColor
        from PyQt5.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QComboBox,
            QDialog,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QSplitter,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
            QHeaderView,
            QCheckBox,
        )
        QT_LIB = 5
    except ImportError:
        _maybe_reexec_with_qt()
        raise SystemExit(
            "No Qt binding found for this interpreter. Install `PyQt5` or `PyQt6`, "
            "or run this script with a Python environment that already provides Qt."
        )

try:
    import nibabel as nib
    import rich  # noqa: F401
except ImportError as exc:
    _maybe_reexec_with_qt()
    raise SystemExit(
        "The GUI interpreter is missing required runtime dependencies. "
        "Activate the mrsiprep conda environment or install nibabel and rich. "
        f"Original import failure: {exc}"
    )


# Embedded CAT12 migration helpers
#!/usr/bin/env python3

import argparse
import os
import re
import shutil
from pathlib import Path


SESSION_FOLDER_PATTERN = re.compile(r"^(?P<sub>.+?)[_-](?P<session>[TtVv]\d+)$")
SUB_SES_PATTERN = re.compile(r"sub-(?P<sub>[^_]+).*ses-(?P<ses>[^_]+)", re.IGNORECASE)
LOOSE_SUB_SES_PATTERN = re.compile(r"(?P<sub>[A-Za-z0-9]+)[_-](?P<ses>[TtVv]\d+)")
CAT12_PREFIX_PATTERN = re.compile(r"^(?P<desc>(?:mwp|wp|p)\d+)", re.IGNORECASE)
MRI_CAT12_ALLOWED_PATTERN = re.compile(r"^(?P<desc>p\d+)_", re.IGNORECASE)


def _emit(log_callback, message):
    if log_callback is not None:
        log_callback(message)


def _build_patterns(input_format, input_pattern):
    if input_pattern:
        return [re.compile(input_pattern, re.IGNORECASE)]

    patterns = []
    if input_format in ("auto", "legacy"):
        patterns.append(
            re.compile(
                r"^"
                r"(?P<desc>p\d+?)"
                r"(?P<sub>\d+)"
                r"_(?P<ses>[TV]\d+)"
                r"_t1_"
                r"(?P<rest>.+?)"
                r"(?P<ext>\.nii(?:\.gz)?)"
                r"$",
                re.IGNORECASE,
            )
        )
    if input_format in ("auto", "simple"):
        patterns.append(
            re.compile(
                r"^"
                r"(?P<desc>p\d+?)"
                r"(?P<sub>[A-Za-z0-9]+)"
                r"_(?P<ses>[TV]\d+)"
                r"_T1w"
                r"(?P<ext>\.nii(?:\.gz)?)"
                r"$",
                re.IGNORECASE,
            )
        )
    if input_format in ("auto", "bids"):
        patterns.append(
            re.compile(
                r"^"
                r"(?P<desc>[A-Za-z0-9]+)"
                r"[_-]?sub-"
                r"(?P<sub>[A-Za-z0-9]+)"
                r"_ses-(?P<ses>[A-Za-z0-9]+)"
                r"(?:_acq-(?P<acq>[A-Za-z0-9]+))?"
                r".*?"
                r"(?P<ext>\.nii(?:\.gz)?)"
                r"$",
                re.IGNORECASE,
            )
        )
    return patterns


def _match_patterns(fname, patterns):
    for pattern in patterns:
        match = pattern.match(fname)
        if match:
            return match
    return None


def _resolve_source_dir(source_dir):
    if os.path.isdir(os.path.join(source_dir, "mri")) and os.path.basename(source_dir) != "mri":
        return os.path.join(source_dir, "mri")
    return source_dir


def _normalize_session(session):
    session = str(session or "").strip()
    if not session:
        return ""
    if session.upper().startswith("T"):
        return f"V{session[1:]}"
    return session


def _strip_results_prefix(name):
    name = str(name or "")
    return name[len("Results_"):] if name.startswith("Results_") else name


def _guess_subject_session_from_path(file_path):
    path = Path(file_path)
    candidates = [path.name, path.stem]
    candidates.extend(parent.name for parent in path.parents)

    for candidate in candidates:
        match = SUB_SES_PATTERN.search(candidate)
        if match:
            return match.group("sub"), _normalize_session(match.group("ses"))

    for parent in path.parents:
        normalized_parent = _strip_results_prefix(parent.name)
        match = SESSION_FOLDER_PATTERN.match(normalized_parent)
        if match:
            return match.group("sub"), _normalize_session(match.group("session"))
        if parent.name.startswith("Results_") and normalized_parent:
            return normalized_parent, ""

    for candidate in candidates:
        normalized_candidate = _strip_results_prefix(candidate)
        match = LOOSE_SUB_SES_PATTERN.search(normalized_candidate)
        if match:
            return match.group("sub"), _normalize_session(match.group("ses"))

    return "", ""


def detect_cat12_file(file_path, input_format="auto", input_pattern=None):
    path = Path(file_path)
    in_mri_folder = any(part.lower() == "mri" for part in path.parts)
    allowed_mri_match = MRI_CAT12_ALLOWED_PATTERN.match(os.path.basename(file_path))

    patterns = _build_patterns(input_format, input_pattern)
    match = _match_patterns(os.path.basename(file_path), patterns)
    if match:
        if in_mri_folder and not allowed_mri_match:
            return None
        ses_raw = match.group("ses").upper()
        ses_id = f"V{ses_raw[1:]}" if ses_raw.startswith("T") else ses_raw
        return {
            "source": file_path,
            "desc": match.group("desc").lower(),
            "sub": match.group("sub"),
            "ses": ses_id,
            "ext": match.groupdict().get("ext"),
            "acq": match.groupdict().get("acq"),
        }

    prefix_match = allowed_mri_match
    if not prefix_match:
        return None

    if not in_mri_folder:
        return None

    sub_id, ses_id = _guess_subject_session_from_path(file_path)
    return {
        "source": file_path,
        "desc": prefix_match.group("desc").lower(),
        "sub": sub_id,
        "ses": ses_id,
        "ext": "".join(path.suffixes[-2:]) if path.name.endswith(".nii.gz") else path.suffix,
        "acq": None,
    }


def iter_cat12_matches(source_dir, input_format="auto", input_pattern=None):
    source_dir = _resolve_source_dir(source_dir)
    if not os.path.isdir(source_dir):
        return []

    matches = []
    for fname in sorted(os.listdir(source_dir)):
        full_path = os.path.join(source_dir, fname)
        if not os.path.isfile(full_path):
            continue
        detected = detect_cat12_file(
            full_path,
            input_format=input_format,
            input_pattern=input_pattern,
        )
        if detected:
            matches.append(detected)
    return matches


def build_cat12_destination(match_info, bids_dir, prefix=None):
    sub_id = match_info["sub"]
    ses_id = match_info["ses"]
    desc = match_info["desc"]
    dest_dir = os.path.join(
        bids_dir,
        "derivatives",
        "cat12",
        f"sub-{sub_id}",
        f"ses-{ses_id}",
    )
    if prefix is not None:
        out_fname = f"sub-{prefix}{sub_id}_ses-{ses_id}_desc-{desc}_T1w.nii.gz"
    else:
        out_fname = f"sub-{sub_id}_ses-{ses_id}_desc-{desc}_T1w.nii.gz"
    return os.path.join(dest_dir, out_fname)


def migrate_cat12_file(match_info, bids_dir, prefix=None, overwrite=False, log_callback=None):
    full_path = match_info["source"]
    fname = os.path.basename(full_path)
    out_path = build_cat12_destination(match_info, bids_dir, prefix=prefix)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not overwrite:
        return out_path, "skipped"

    if fname.lower().endswith(".nii") and not fname.lower().endswith(".nii.gz"):
        img = nib.load(full_path)
        nib.save(img, out_path)
        _emit(log_callback, f"Converted: {fname} -> {os.path.basename(out_path)}")
        return out_path, "converted"
    if fname.lower().endswith(".nii.gz"):
        shutil.copy2(full_path, out_path)
        _emit(log_callback, f"Copied: {fname} -> {os.path.basename(out_path)}")
        return out_path, "copied"
    raise ValueError(f"Skipping '{fname}': unsupported extension")


def migrate_cat12_dir(
    source_dir,
    bids_dir,
    input_format="auto",
    input_pattern=None,
    prefix=None,
    overwrite=False,
    log_callback=print,
):
    matches = iter_cat12_matches(
        source_dir,
        input_format=input_format,
        input_pattern=input_pattern,
    )
    summary = {"files": [], "skipped": [], "errors": []}
    for match_info in matches:
        try:
            out_path, status = migrate_cat12_file(
                match_info,
                bids_dir,
                prefix=prefix,
                overwrite=overwrite,
                log_callback=log_callback,
            )
            if status != "skipped":
                summary["files"].append(out_path)
        except Exception as exc:
            summary["errors"].append(f"Failed processing {match_info['source']}: {exc}")

    source_dir = _resolve_source_dir(source_dir)
    if not matches:
        for fname in sorted(os.listdir(source_dir)):
            full_path = os.path.join(source_dir, fname)
            if os.path.isfile(full_path):
                summary["skipped"].append(f"Skipping '{fname}': pattern mismatch")
    return summary


def main(
    source_dir,
    bids_dir,
    input_format="auto",
    input_pattern=None,
    prefix=None,
    overwrite=False,
):
    return migrate_cat12_dir(
        source_dir,
        bids_dir,
        input_format=input_format,
        input_pattern=input_pattern,
        prefix=prefix,
        overwrite=overwrite,
        log_callback=print,
    )

# Embedded MRSI LCModel migration helpers
import glob
import os
import re

import nibabel as nib

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Include patterns to filter files (exact substrings as provided).
# include_metabolite_list = [
#     "FWHM", "SNR", "NAA", "NAAG", "Ins", "GPC+PCh", "Glu",
#     "Gln", "Cr+PCr", "GABA", "GSH", "NAA+NAAG", "Glu+Gln",
# ]
include_metabolite_list = [
    "FWHM",
    "SNR",
    "Ins",
    "GPC+PCh",
    "Cr+PCr",
    "NAA+NAAG",
    "Glu+Gln",
]
include_others_list = ["FWHM", "SNR", "brainmask", "WaterSignal"]
DEFAULT_MRSI_METABOLITES = include_metabolite_list
DEFAULT_MRSI_OTHERS = include_others_list


# Mapping for acq values to new descriptor names for non-metabolite files.
ACQ_MAPPING = {
    "conc": "signal",
    "crlb": "crlb",
    "snr": "snr",
    "fwhm": "fwhm",
}
SESSION_PATTERN = re.compile(r"^(?P<sub>.+?)[_-](?P<session>[TtVv]\d+)$")


def _emit(log_callback, message):
    if log_callback is not None:
        log_callback(message)


def parse_subject_session_folder(folder_name):
    normalized_name = str(folder_name or "")
    if normalized_name.startswith("Results_"):
        normalized_name = normalized_name[len("Results_"):]
    match = SESSION_PATTERN.match(normalized_name)
    if not match:
        return None
    session = match.group("session")
    return {
        "sub": match.group("sub"),
        "session": session,
        "new_session": f"V{session[1:]}",
    }


def _infer_anat_acq(filename):
    lower = filename.lower()
    if "mp2rage" in lower:
        return "mp2rage"
    if "mprage" in lower:
        return "mprage"
    return "unknown"


def find_mrsi_nifti_dir(folder_path, input_subdir=None):
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    candidates = []
    candidates.append(os.path.join(folder_path, "MRSI_NIFTI"))
    if input_subdir:
        candidates.append(
            os.path.join(folder_path, input_subdir, f"Results_{folder_name}", "MRSI_NIFTI")
        )
        candidates.append(os.path.join(folder_path, input_subdir, "MRSI_NIFTI"))
    candidates.append(os.path.join(folder_path, f"Results_{folder_name}", "MRSI_NIFTI"))
    return next((candidate for candidate in candidates if os.path.isdir(candidate)), None)


def find_matching_mrsi_files(mrs_dir, include_metabolites=None):
    if include_metabolites is None:
        include_metabolites = include_metabolite_list
    all_files = glob.glob(os.path.join(mrs_dir, "*.nii"))
    all_files.extend(glob.glob(os.path.join(mrs_dir, "*.nii.gz")))
    filtered_files = []
    seen = set()
    for file_path in all_files:
        filename = os.path.basename(file_path)
        for inc_pattern in include_metabolites:
            if f"_{inc_pattern}_" in filename and file_path not in seen:
                filtered_files.append(file_path)
                seen.add(file_path)
        for inc_pattern in include_others_list:
            if f"_{inc_pattern}." in filename and file_path not in seen:
                filtered_files.append(file_path)
                seen.add(file_path)
    return filtered_files


def find_anatomical_candidates(folder_path):
    anat_candidates = []
    for pattern in ("*.nii", "*.nii.gz"):
        anat_candidates.extend(glob.glob(os.path.join(folder_path, pattern)))
    return sorted(set(anat_candidates))


def migrate_anatomical_file(
    anat_path,
    dest_bids_dir,
    sub,
    session,
    overwrite=False,
    log_callback=None,
    acq_override=None,
):
    if not anat_path.endswith((".nii", ".nii.gz")):
        raise ValueError(f"Skipping non-NIfTI file: {anat_path}")

    acq = acq_override or _infer_anat_acq(os.path.basename(anat_path))
    anat_dir = os.path.join(dest_bids_dir, f"sub-{sub}", f"ses-{session}", "anat")
    anat_name = f"sub-{sub}_ses-{session}_acq-{acq}_T1w.nii.gz"
    anat_full = os.path.join(anat_dir, anat_name)

    os.makedirs(anat_dir, exist_ok=True)
    if os.path.exists(anat_full) and not overwrite:
        return anat_full, "skipped"

    img = nib.load(anat_path)
    nib.save(img, anat_full)
    _emit(log_callback, f"Copying anatomical file {anat_path} to {anat_full}")
    return anat_full, "copied"


def parse_mrsi_filename(old_path):
    fname = os.path.basename(old_path)
    new_met = None
    new_desc = None

    if re.search(r"brain[_]?mask", fname, re.IGNORECASE):
        new_desc = "brainmask"
    elif re.search(r"watersignal", fname, re.IGNORECASE):
        new_met = "water"
        new_desc = "signal"
    else:
        base = fname[:-7] if fname.endswith(".nii.gz") else fname
        if base.startswith("OrigRes_"):
            base = base[len("OrigRes_"):]
        parts = [part for part in base.split("_") if part]
        if not parts:
            raise ValueError(f"Skipping file (unexpected format): {fname}")

        if len(parts) == 1:
            acq = parts[0].lower()
            if acq not in ACQ_MAPPING:
                if "metabnorm" in acq:
                    raise RuntimeError(f"Skipping unsupported metabnorm file: {fname}")
                raise KeyError(f"Unknown acq value '{acq}' in file {fname}. Skipping.")
            new_desc = ACQ_MAPPING[acq]
        else:
            met = parts[0]
            acq = parts[1].lower()
            if acq not in ACQ_MAPPING:
                tail_acq = "_".join(parts[1:]).lower()
                if tail_acq in ACQ_MAPPING:
                    acq = tail_acq
                else:
                    if "metabnorm" in acq or "metabnorm" in tail_acq:
                        raise RuntimeError(f"Skipping unsupported metabnorm file: {fname}")
                    raise KeyError(f"Unknown acq value '{acq}' in file {fname}. Skipping.")
            new_desc = ACQ_MAPPING[acq]
            cleaned_met = met.replace("+", "").strip()
            if cleaned_met and cleaned_met.lower() != "voxel":
                new_met = cleaned_met

    return {"met": new_met, "desc": new_desc}


def _build_mrsi_destination(old_path, dest_bids_dir, sub, session, met_override=None, desc_override=None):
    parsed = parse_mrsi_filename(old_path)
    if met_override is None:
        new_met = parsed["met"]
    else:
        new_met = str(met_override).strip() or None
    if desc_override is None:
        new_desc = parsed["desc"]
    else:
        new_desc = str(desc_override).strip() or parsed["desc"]

    if not new_desc:
        raise ValueError(f"Could not determine desc for {old_path}")

    new_name = f"sub-{sub}_ses-{session}_space-orig"
    if new_met is not None:
        new_name += f"_met-{new_met}"
    new_name += f"_desc-{new_desc}_mrsi.nii.gz"
    new_dir = os.path.join(dest_bids_dir, "derivatives", "mrsi-orig", f"sub-{sub}", f"ses-{session}")
    return os.path.join(new_dir, new_name)


def build_mrsi_destination(old_path, dest_bids_dir, sub, session, met_override=None, desc_override=None):
    return _build_mrsi_destination(
        old_path,
        dest_bids_dir,
        sub,
        session,
        met_override=met_override,
        desc_override=desc_override,
    )


def migrate_mrsi_file(
    old_path,
    dest_bids_dir,
    sub,
    session,
    overwrite=False,
    log_callback=None,
    met_override=None,
    desc_override=None,
):
    new_full = build_mrsi_destination(
        old_path,
        dest_bids_dir,
        sub,
        session,
        met_override=met_override,
        desc_override=desc_override,
    )
    os.makedirs(os.path.dirname(new_full), exist_ok=True)
    if os.path.exists(new_full) and not overwrite:
        return new_full, "skipped"
    img = nib.load(old_path)
    nib.save(img, new_full)
    _emit(log_callback, f"Copying MRSI file {old_path} to {new_full}")
    return new_full, "copied"


def migrate_session_folder(
    folder_path,
    dest_bids_dir,
    input_subdir=None,
    migrate_anat=False,
    overwrite=False,
    log_callback=None,
    subject_override=None,
    session_override=None,
    include_metabolites=None,
):
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    meta = parse_subject_session_folder(folder_name)
    summary = {
        "folder": folder_path,
        "subject": None,
        "session": None,
        "mrsi_files": [],
        "anat_files": [],
        "skipped": [],
        "errors": [],
    }

    if not meta:
        summary["skipped"].append(f"Skipping folder (does not match pattern): {folder_name}")
        return summary

    sub = subject_override or meta["sub"]
    session = session_override or meta["new_session"]
    if session and not str(session).upper().startswith("V"):
        session = f"V{str(session).lstrip('tTvV')}"
    summary["subject"] = sub
    summary["session"] = session

    mrs_dir = find_mrsi_nifti_dir(folder_path, input_subdir=input_subdir)
    if not mrs_dir:
        summary["skipped"].append(f"Skipping MRSI, MRSI_NIFTI not found under: {folder_path}")
    else:
        filtered_files = find_matching_mrsi_files(mrs_dir, include_metabolites=include_metabolites)
        _emit(
            log_callback,
            f"Processing folder {folder_path} (subject: {sub}, session: {session}), found {len(filtered_files)} matching files.",
        )
        for old_path in filtered_files:
            new_full = None
            try:
                new_full, status = migrate_mrsi_file(
                    old_path,
                    dest_bids_dir,
                    sub,
                    session,
                    overwrite=overwrite,
                    log_callback=log_callback,
                )
                if status != "skipped":
                    summary["mrsi_files"].append(new_full)
            except RuntimeError:
                continue
            except (KeyError, ValueError) as exc:
                summary["skipped"].append(str(exc))
            except Exception as exc:
                summary["errors"].append(f"Error copying file {old_path} to {new_full or 'unknown'}: {exc}")

    if migrate_anat:
        for anat_path in find_anatomical_candidates(folder_path):
            try:
                anat_full, status = migrate_anatomical_file(
                    anat_path,
                    dest_bids_dir,
                    sub,
                    session,
                    overwrite=overwrite,
                    log_callback=log_callback,
                )
                if status == "copied":
                    summary["anat_files"].append(anat_full)
            except Exception as exc:
                summary["errors"].append(f"Error copying anatomical file {anat_path}: {exc}")

    return summary


def rename_mrsi_files(
    source_dir,
    dest_bids_dir,
    input_subdir=None,
    migrate_anat=False,
    overwrite=False,
):
    """
    Process each subject folder in source_dir with the structure:
      SOURCE_DIR/SUBJECTID_Ti/Results_SUBJECTID_Ti/MRSI_NIFTI/*.nii.gz
    """
    console = Console()
    folder_names = [
        folder
        for folder in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, folder))
    ]

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(*progress_columns, console=console) as progress:
        folders_task = progress.add_task("Folders", total=len(folder_names))
        for folder in folder_names:
            folder_path = os.path.join(source_dir, folder)
            meta = parse_subject_session_folder(folder)
            summary = migrate_session_folder(
                folder_path,
                dest_bids_dir,
                input_subdir=input_subdir,
                migrate_anat=migrate_anat,
                overwrite=overwrite,
                log_callback=console.print,
            )

            if meta:
                mrsi_total = len(find_matching_mrsi_files(find_mrsi_nifti_dir(folder_path, input_subdir) or "")) if find_mrsi_nifti_dir(folder_path, input_subdir) else 0
                files_task = progress.add_task(
                    f"Files {meta['sub']} {meta['new_session']}",
                    total=max(mrsi_total, 1),
                )
                progress.advance(files_task, max(mrsi_total, 1))
                progress.remove_task(files_task)

                if migrate_anat:
                    anat_total = len(find_anatomical_candidates(folder_path))
                    anat_task = progress.add_task(
                        f"Anat {meta['sub']} {meta['new_session']}",
                        total=max(anat_total, 1),
                    )
                    progress.advance(anat_task, max(anat_total, 1))
                    progress.remove_task(anat_task)

            for message in summary["skipped"] + summary["errors"]:
                console.print(message)

            progress.advance(folders_task)

try:
    from run_hd_bet_batch import process_t1_files
except Exception as exc:
    _maybe_reexec_with_qt()
    raise SystemExit(
        "The GUI interpreter could not import scripts/run_hd_bet_batch.py. "
        f"Original import failure: {exc}"
    )

# End embedded migration helpers


def import_mridata():
    try:
        from tools.mridata import MRIData  # type: ignore
        return MRIData.extract_metadata
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        try:
            sys.path.append(str(repo_root))
            from tools.mridata import MRIData  # type: ignore
            return MRIData.extract_metadata
        except Exception:
            def _fallback_extract_metadata(filename):
                base = os.path.basename(str(filename))
                metadata = {}
                for key in ("sub", "ses", "acq"):
                    match = re.search(rf"{key}-([^_]+)", base)
                    metadata[key] = match.group(1) if match else None
                return metadata
            return _fallback_extract_metadata


extract_metadata = import_mridata()

ROLE_IGNORE = "Ignore"
ROLE_MRSI = "MRSI File"
ROLE_T1 = "T1 Image"
ROLE_CAT12 = "CAT12 File"
ROLE_OPTIONS = [ROLE_IGNORE, ROLE_MRSI, ROLE_T1, ROLE_CAT12]
CAT12_PREFIX_PATTERN = re.compile(r"^(?P<desc>(?:mwp|wp|p)\d+)", re.IGNORECASE)
SUB_SES_PATTERN = re.compile(r"sub-(?P<sub>[^_]+).*ses-(?P<ses>[^_]+)", re.IGNORECASE)
LOOSE_SUB_SES_PATTERN = re.compile(r"(?P<sub>[A-Za-z0-9]+)[_-](?P<ses>[TtVv]\d+)")
DEFAULT_IGNORE_PATH_SUBSTRINGS = [
    "MetabNorm",
    "thalamic",
    "thalamus",
    "lpba",
    "_bet",
    "y_",
    "suit_",
    "cat_",
    "julich",
    "cobra",
    "aal3",
    "neuromorph",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
    "m7",
    "m8",
    "m9",
    "rh.",
    "lh.",
]
MRSI_7T_METABOLITES = list(
    dict.fromkeys(
        [
            *DEFAULT_MRSI_METABOLITES,
            "NAA",
            "NAAG",
            "Glu",
            "Gln",
            "GABA",
            "GSH",
        ]
    )
)


@dataclass
class DetectedItem:
    source_path: str
    detected_kind: str
    default_role: str
    subject: str = ""
    session: str = ""
    acq: str = ""
    met: str = ""
    desc: str = ""
    notes: str = ""
    confidence: str = "medium"
    cat12_match: dict = field(default_factory=dict)
    mrsi_file_count: int = 0


def is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def normalize_session(session: str) -> str:
    session = str(session or "").strip()
    if not session:
        return ""
    if session.upper().startswith("V"):
        return f"V{session[1:]}"
    if session.upper().startswith("T"):
        return f"V{session[1:]}"
    return session


def apply_subject_prefix(subject: str, prefix: str) -> str:
    subject = str(subject or "").strip()
    prefix = str(prefix or "").strip()
    if not subject:
        return ""
    return f"{prefix}{subject}" if prefix else subject


def _user_role():
    return Qt.ItemDataRole.UserRole if QT_LIB == 6 else Qt.UserRole


def _path_display_name(path: str) -> str:
    return Path(path).name


def _set_path_item_display(item, value: str, is_path: bool):
    value = str(value or "")
    if is_path and value:
        item.setText(_path_display_name(value))
        item.setToolTip(value)
        item.setData(_user_role(), value)
    else:
        item.setText(value)
        item.setToolTip("")
        item.setData(_user_role(), None)


def _set_noneditable(item):
    if QT_LIB == 6:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    else:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)


def _center_alignment():
    return Qt.AlignmentFlag.AlignCenter if QT_LIB == 6 else Qt.AlignCenter


def _set_exists_indicator(item, target: str, target_is_path: bool):
    item.setText("")
    if not target_is_path or not target or target.startswith("<"):
        item.setData(_user_role(), 2)
        item.setToolTip("")
        item.setBackground(QColor("#6b7280"))
        return
    if os.path.exists(target):
        item.setData(_user_role(), 1)
        item.setToolTip("Target exists")
        item.setBackground(QColor("#16a34a"))
    else:
        item.setData(_user_role(), 0)
        item.setToolTip("Target does not exist")
        item.setBackground(QColor("#dc2626"))


def parse_ignore_terms(raw_text: str):
    terms = []
    for chunk in re.split(r"[\n,;]+", str(raw_text or "")):
        token = chunk.strip()
        if not token:
            continue
        terms.append(token.lower())
    return terms


def parse_csv_terms(raw_text: str):
    terms = []
    for chunk in re.split(r"[\n,;]+", str(raw_text or "")):
        token = chunk.strip()
        if token:
            terms.append(token)
    return terms


def should_ignore_path(path, ignore_terms) -> bool:
    if not ignore_terms:
        return False
    haystack = str(path).lower()
    return any(term in haystack for term in ignore_terms)


def mrsi_filename_matches_selection(path: Path, include_metabolites=None) -> bool:
    include_metabolites = DEFAULT_MRSI_METABOLITES if include_metabolites is None else include_metabolites
    filename = path.name
    for inc_pattern in include_metabolites:
        if f"_{inc_pattern}_" in filename:
            return True
    for inc_pattern in DEFAULT_MRSI_OTHERS:
        if f"_{inc_pattern}." in filename:
            return True
    return False


def safe_extract_metadata(path: Path):
    try:
        return extract_metadata(str(path))
    except Exception:
        return {}


def guess_subject_session_from_path(path: Path, subject_override="", session_override="", use_metadata=True):
    subject = subject_override or ""
    session = normalize_session(session_override)
    note_parts = []

    metadata = safe_extract_metadata(path) if use_metadata else {}
    if metadata:
        if not subject and metadata.get("sub"):
            subject = metadata.get("sub") or ""
            note_parts.append("subject from filename")
        if not session and metadata.get("ses"):
            session = normalize_session(metadata.get("ses"))
            note_parts.append("session from filename")

    if not subject or not session:
        for ancestor in path.parents:
            bids_match = SUB_SES_PATTERN.search(ancestor.name)
            if bids_match:
                if not subject:
                    subject = bids_match.group("sub")
                    note_parts.append(f"subject from folder {ancestor.name}")
                if not session:
                    session = normalize_session(bids_match.group("ses"))
                    note_parts.append(f"session from folder {ancestor.name}")
                break
            if ancestor.name.startswith("Results_") and not subject:
                results_name = ancestor.name[len("Results_"):]
                parsed_results = parse_subject_session_folder(results_name)
                if parsed_results:
                    subject = parsed_results["sub"]
                    note_parts.append(f"subject from folder {ancestor.name}")
                    if not session:
                        session = parsed_results["new_session"]
                        note_parts.append(f"session from folder {ancestor.name}")
                    break
                if results_name:
                    subject = results_name
                    note_parts.append(f"subject from folder {ancestor.name}")
            parsed = parse_subject_session_folder(ancestor.name)
            if not parsed:
                continue
            if not subject:
                subject = parsed["sub"]
                note_parts.append(f"subject from folder {ancestor.name}")
            if not session:
                session = parsed["new_session"]
                note_parts.append(f"session from folder {ancestor.name}")
            break

    if not subject or not session:
        for candidate in (path.stem, path.name):
            bids_match = SUB_SES_PATTERN.search(candidate)
            if bids_match:
                if not subject:
                    subject = bids_match.group("sub")
                    note_parts.append("subject from BIDS pattern")
                if not session:
                    session = normalize_session(bids_match.group("ses"))
                    note_parts.append("session from BIDS pattern")
                break
            match = LOOSE_SUB_SES_PATTERN.search(candidate)
            if not match:
                continue
            if not subject:
                subject = match.group("sub")
                note_parts.append("subject from loose pattern")
            if not session:
                session = normalize_session(match.group("ses"))
                note_parts.append("session from loose pattern")
            break

    return subject, session, ", ".join(note_parts)


def guess_subject_session_for_t1(path: Path, subject_override="", session_override=""):
    subject, session, path_note = guess_subject_session_from_path(
        path,
        subject_override=subject_override,
        session_override=session_override,
    )
    acq = _infer_anat_acq(path.name)
    metadata = safe_extract_metadata(path)
    if metadata and metadata.get("acq"):
        acq = metadata.get("acq") or acq
    return subject, session, acq, path_note


def is_probable_cat12(path: Path) -> bool:
    if not is_nifti_path(path):
        return False
    if not any(part.lower() == "mri" for part in path.parts):
        return False
    return CAT12_PREFIX_PATTERN.match(path.name) is not None


def is_probable_t1(path: Path) -> bool:
    if not is_nifti_path(path):
        return False
    lower = path.name.lower()
    positive_tokens = ("t1w", "_t1", "mprage", "mp2rage", "memprage")
    negative_tokens = (
        "brainmask",
        "desc-brain",
        "watersignal",
        "origres",
        "mrsi",
        "crlb",
        "fwhm",
        "snr",
        "gpc+pch",
        "glu+gln",
        "cr+pcr",
        "naa+naag",
        "mwp",
    )
    if is_probable_cat12(path):
        return False
    if any(token in lower for token in negative_tokens):
        return False
    return any(token in lower for token in positive_tokens)


def build_mrsi_item(session_path: Path, input_subdir=None, include_metabolites=None):
    meta = parse_subject_session_folder(session_path.name)
    if not meta:
        return None
    mrs_dir = find_mrsi_nifti_dir(str(session_path), input_subdir=input_subdir)
    if not mrs_dir:
        return None
    file_count = len(find_matching_mrsi_files(mrs_dir, include_metabolites=include_metabolites))
    notes = f"MRSI session with {file_count} matching outputs"
    return DetectedItem(
        source_path=str(session_path),
        detected_kind=ROLE_MRSI,
        default_role=ROLE_MRSI,
        subject=meta["sub"],
        session=meta["new_session"],
        notes=notes,
        confidence="high",
        mrsi_file_count=file_count,
    )


def is_probable_mrsi_file(path: Path) -> bool:
    if not is_nifti_path(path):
        return False
    lower = path.name.lower()
    return "origres" in lower or any(part.upper() == "MRSI_NIFTI" for part in path.parts)


def build_mrsi_file_item(
    path: Path,
    subject_override="",
    session_override="",
    force=False,
    note_prefix="",
    include_metabolites=None,
):
    if not force and not is_probable_mrsi_file(path):
        return None
    if not force and not mrsi_filename_matches_selection(path, include_metabolites=include_metabolites):
        return None
    subject, session, guess_note = guess_subject_session_from_path(
        path,
        subject_override=subject_override,
        session_override=session_override,
        use_metadata=False,
    )
    try:
        parsed = parse_mrsi_filename(str(path))
    except Exception:
        parsed = {"met": "", "desc": ""}
    note_parts = [note_prefix] if note_prefix else []
    note_parts.append("OrigRes MRSI file" if "origres" in path.name.lower() else "MRSI file")
    if guess_note:
        note_parts.append(guess_note)
    if not subject or not session:
        note_parts.append("edit subject/session before running")
    if parsed.get("desc"):
        note_parts.append(f"desc {parsed['desc']}")
    if parsed.get("met"):
        note_parts.append(f"met {parsed['met']}")
    return DetectedItem(
        source_path=str(path),
        detected_kind=ROLE_MRSI,
        default_role=ROLE_MRSI,
        subject=subject,
        session=session,
        met=parsed.get("met") or "",
        desc=parsed.get("desc") or "",
        notes=", ".join(part for part in note_parts if part),
        confidence="medium" if subject and session else "low",
    )


def build_t1_item(path: Path, subject_override="", session_override="", force=False, note_prefix=""):
    if not force and not is_probable_t1(path):
        return None
    subject, session, acq, guess_note = guess_subject_session_for_t1(
        path,
        subject_override=subject_override,
        session_override=session_override,
    )
    note_parts = [note_prefix] if note_prefix else []
    if guess_note:
        note_parts.append(guess_note)
    if not subject or not session:
        note_parts.append("edit subject/session before running")
    return DetectedItem(
        source_path=str(path),
        detected_kind=ROLE_T1,
        default_role=ROLE_T1,
        subject=subject,
        session=session,
        acq=acq,
        notes=", ".join(part for part in note_parts if part),
        confidence="medium" if subject and session else "low",
    )


def build_cat12_item(path: Path, input_format="auto", input_pattern=None):
    match_info = detect_cat12_file(
        str(path),
        input_format=input_format,
        input_pattern=input_pattern,
    )
    if not match_info:
        return None
    note_parts = [f"CAT12 desc-{match_info['desc']}"]
    if any(part.lower() == "mri" for part in path.parts):
        note_parts.append("from mri folder")
    if not match_info.get("sub") or not match_info.get("ses"):
        note_parts.append("edit subject/session before running")
    return DetectedItem(
        source_path=str(path),
        detected_kind=ROLE_CAT12,
        default_role=ROLE_CAT12,
        subject=match_info.get("sub") or "",
        session=normalize_session(match_info.get("ses") or ""),
        acq=match_info.get("acq") or "",
        notes=", ".join(note_parts),
        confidence="high" if match_info.get("sub") and match_info.get("ses") else "medium",
        cat12_match=match_info,
    )


def collect_nifti_files(source_paths, ignore_terms=None):
    nifti_files = []
    seen = set()
    skip_dirs = {"derivatives", ".git", "__pycache__"}

    def add_file(path):
        if should_ignore_path(path, ignore_terms):
            return
        key = os.path.realpath(str(path))
        if key in seen:
            return
        seen.add(key)
        nifti_files.append(Path(path))

    for raw_path in source_paths:
        source = Path(raw_path).expanduser()
        if not source.exists():
            continue
        if should_ignore_path(source, ignore_terms):
            continue
        if source.is_file():
            if is_nifti_path(source):
                add_file(source)
            continue

        for current_root, dirnames, filenames in os.walk(source):
            if should_ignore_path(current_root, ignore_terms):
                dirnames[:] = []
                continue
            dirnames[:] = [name for name in dirnames if name not in skip_dirs]
            dirnames[:] = [
                name
                for name in dirnames
                if not should_ignore_path(Path(current_root) / name, ignore_terms)
            ]
            for filename in sorted(filenames):
                file_path = Path(current_root) / filename
                if is_nifti_path(file_path):
                    add_file(file_path)

    return sorted(nifti_files, key=lambda path: str(path))


def create_detection_for_role(
    path: Path,
    role: str,
    subject="",
    session="",
    acq="",
    met="",
    desc="",
    cat12_format="auto",
    cat12_pattern=None,
):
    if role == ROLE_T1:
        item = build_t1_item(
            path,
            subject_override=subject,
            session_override=session,
            force=True,
            note_prefix="manual selection",
        )
        if item is None:
            item = DetectedItem(str(path), ROLE_T1, ROLE_T1, notes="manual selection", confidence="manual")
        item.acq = acq or item.acq or _infer_anat_acq(path.name)
    elif role == ROLE_CAT12:
        item = build_cat12_item(
            path,
            input_format=cat12_format,
            input_pattern=cat12_pattern,
        )
        if item is None:
            desc_match = CAT12_PREFIX_PATTERN.match(path.name)
            desc = desc_match.group("desc").lower() if desc_match else "p0"
            item = DetectedItem(
                str(path),
                ROLE_CAT12,
                ROLE_CAT12,
                notes="manual selection",
                confidence="manual",
                cat12_match={"source": str(path), "desc": desc, "sub": subject, "ses": normalize_session(session), "acq": ""},
            )
        item.cat12_match = dict(item.cat12_match or {})
        item.cat12_match["source"] = str(path)
        item.cat12_match["desc"] = item.cat12_match.get("desc") or "p0"
    elif role == ROLE_MRSI:
        item = build_mrsi_file_item(
            path,
            subject_override=subject,
            session_override=session,
            force=True,
            note_prefix="manual selection",
        )
        if item is None:
            item = DetectedItem(str(path), ROLE_MRSI, ROLE_MRSI, notes="manual selection", confidence="manual")
    else:
        item = DetectedItem(str(path), role, role, notes="manual selection", confidence="manual")

    if subject:
        item.subject = subject
    if session:
        item.session = normalize_session(session)
    if role == ROLE_T1 and acq:
        item.acq = acq
    if role == ROLE_MRSI:
        if met:
            item.met = met
        if desc:
            item.desc = desc
    if role == ROLE_CAT12:
        item.cat12_match["sub"] = item.subject
        item.cat12_match["ses"] = item.session
    return item


def scan_input_paths(
    source_paths,
    input_subdir=None,
    cat12_format="auto",
    cat12_pattern=None,
    ignore_terms=None,
    include_metabolites=None,
):
    detections = []
    seen_paths = set()
    skip_dirs = {"derivatives", ".git", "__pycache__"}

    def add_detection(item):
        if item is None:
            return
        if should_ignore_path(item.source_path, ignore_terms):
            return
        key = os.path.realpath(item.source_path)
        if key in seen_paths:
            return
        seen_paths.add(key)
        detections.append(item)

    for raw_path in source_paths:
        source = Path(raw_path).expanduser()
        if not source.exists():
            continue
        if should_ignore_path(source, ignore_terms):
            continue
        if source.is_file():
            add_detection(build_cat12_item(source, input_format=cat12_format, input_pattern=cat12_pattern))
            if os.path.realpath(str(source)) not in seen_paths and is_nifti_path(source):
                add_detection(build_mrsi_file_item(source, include_metabolites=include_metabolites))
            if os.path.realpath(str(source)) not in seen_paths and is_nifti_path(source):
                add_detection(build_t1_item(source))
            continue

        for current_root, dirnames, filenames in os.walk(source):
            if should_ignore_path(current_root, ignore_terms):
                dirnames[:] = []
                continue
            dirnames[:] = [name for name in dirnames if name not in skip_dirs]
            dirnames[:] = [
                name
                for name in dirnames
                if not should_ignore_path(Path(current_root) / name, ignore_terms)
            ]
            current_path = Path(current_root)

            mrsi_item = build_mrsi_item(
                current_path,
                input_subdir=input_subdir,
                include_metabolites=include_metabolites,
            )
            if mrsi_item is not None:
                mrs_dir = find_mrsi_nifti_dir(str(current_path), input_subdir=input_subdir)
                mrsi_files = (
                    find_matching_mrsi_files(mrs_dir, include_metabolites=include_metabolites)
                    if mrs_dir
                    else []
                )
                for mrsi_path in mrsi_files:
                    add_detection(
                        build_mrsi_file_item(
                            Path(mrsi_path),
                            subject_override=mrsi_item.subject,
                            session_override=mrsi_item.session,
                            force=True,
                            note_prefix="from MRSI_NIFTI",
                        )
                    )
                for anat_path in find_anatomical_candidates(str(current_path)):
                    add_detection(
                        build_t1_item(
                            Path(anat_path),
                            subject_override=mrsi_item.subject,
                            session_override=mrsi_item.session,
                            force=True,
                            note_prefix="top-level anatomical file next to MRSI output",
                        )
                    )
                dirnames[:] = [name for name in dirnames if name.lower() == "mri"]
                continue

            for filename in sorted(filenames):
                file_path = current_path / filename
                if not is_nifti_path(file_path):
                    continue
                add_detection(
                    build_cat12_item(
                        file_path,
                        input_format=cat12_format,
                        input_pattern=cat12_pattern,
                    )
                )
                if os.path.realpath(str(file_path)) not in seen_paths:
                    add_detection(build_mrsi_file_item(file_path, include_metabolites=include_metabolites))
                if os.path.realpath(str(file_path)) not in seen_paths:
                    add_detection(build_t1_item(file_path))

    detections.sort(key=lambda item: (item.detected_kind, item.subject, item.session, item.source_path))
    return detections


class ManualSelectionDialog(QDialog):
    def __init__(
        self,
        source_paths,
        cat12_format="auto",
        cat12_pattern=None,
        ignore_terms=None,
        include_metabolites=None,
        parent=None,
    ):
        super().__init__(parent)
        self.source_paths = source_paths
        self.cat12_format = cat12_format
        self.cat12_pattern = cat12_pattern
        self.ignore_terms = ignore_terms or []
        self.include_metabolites = include_metabolites
        self._loading = False
        self._files = collect_nifti_files(source_paths, ignore_terms=self.ignore_terms)
        self._build_ui()
        self._populate_table()

    def _build_ui(self):
        self.setWindowTitle("Manual File Selection")
        self.resize(1300, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel(
            "No reliable auto-detection found. Select the files you want to add, assign each one a role, and edit subject/session if needed."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Filter:"))
        self.search_edit = QLineEdit("")
        self.search_edit.setPlaceholderText("Substring or regex. Prefix with re: for regex.")
        self.search_edit.textChanged.connect(self._apply_filter)
        search_layout.addWidget(self.search_edit, 1)
        self.select_visible_button = QPushButton("Select Visible")
        self.select_visible_button.clicked.connect(lambda: self._set_visible_checked(True))
        search_layout.addWidget(self.select_visible_button)
        self.clear_visible_button = QPushButton("Clear Visible")
        self.clear_visible_button.clicked.connect(lambda: self._set_visible_checked(False))
        search_layout.addWidget(self.clear_visible_button)
        layout.addLayout(search_layout)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Add", "Role", "Source", "Subject", "Session", "Acq", "Notes"])
        header = self.table.horizontalHeader()
        if QT_LIB == 6:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        else:
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        self.apply_button = QPushButton("Add Selected Files")
        self.apply_button.clicked.connect(self.accept)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

    def _autodetect_role(self, path: Path):
        item = build_cat12_item(path, input_format=self.cat12_format, input_pattern=self.cat12_pattern)
        if item is not None:
            return item
        item = build_mrsi_file_item(path, include_metabolites=self.include_metabolites)
        if item is not None:
            return item
        item = build_t1_item(path)
        if item is not None:
            return item
        return DetectedItem(str(path), ROLE_IGNORE, ROLE_IGNORE, notes="manual review needed", confidence="low")

    def _make_source_item(self, path: Path):
        item = QTableWidgetItem(_path_display_name(str(path)))
        item.setToolTip(str(path))
        item.setData(_user_role(), str(path))
        if QT_LIB == 6:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        else:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _populate_table(self):
        self._loading = True
        self.table.setRowCount(len(self._files))
        for row, path in enumerate(self._files):
            detection = self._autodetect_role(path)

            check = QCheckBox()
            check.setChecked(detection.default_role != ROLE_IGNORE)
            self.table.setCellWidget(row, 0, check)

            role_combo = QComboBox()
            role_combo.addItems(ROLE_OPTIONS)
            role_combo.setCurrentText(detection.default_role)
            role_combo.currentTextChanged.connect(lambda value, r=row: self._on_role_changed(r, value))
            self.table.setCellWidget(row, 1, role_combo)

            self.table.setItem(row, 2, self._make_source_item(path))

            self.table.setItem(row, 3, QTableWidgetItem(detection.subject))
            self.table.setItem(row, 4, QTableWidgetItem(detection.session))
            self.table.setItem(row, 5, QTableWidgetItem(detection.acq))

            notes_text = detection.notes
            if detection.confidence:
                notes_text = f"{notes_text} [{detection.confidence}]".strip() if notes_text else f"[{detection.confidence}]"
            notes_item = QTableWidgetItem(notes_text)
            if QT_LIB == 6:
                notes_item.setFlags(notes_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            else:
                notes_item.setFlags(notes_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6, notes_item)
        self._loading = False
        self._apply_filter()

    def _on_role_changed(self, row, value):
        if self._loading:
            return
        checkbox = self.table.cellWidget(row, 0)
        if checkbox is not None:
            checkbox.setChecked(value != ROLE_IGNORE)

    def _match_filter(self, text, pattern):
        if not pattern:
            return True
        if pattern.startswith("re:"):
            try:
                return re.search(pattern[3:], text, re.IGNORECASE) is not None
            except re.error:
                return True
        return pattern.lower() in text.lower()

    def _apply_filter(self):
        pattern = self.search_edit.text().strip()
        for row in range(self.table.rowCount()):
            haystack = " | ".join(
                [
                    self.table.item(row, 2).toolTip() or self.table.item(row, 2).text(),
                    self.table.item(row, 6).text(),
                ]
            )
            self.table.setRowHidden(row, not self._match_filter(haystack, pattern))

    def _set_visible_checked(self, checked):
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            checkbox = self.table.cellWidget(row, 0)
            if checkbox is not None:
                checkbox.setChecked(checked)

    def selected_items(self):
        items = []
        for row in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(row, 0)
            role_combo = self.table.cellWidget(row, 1)
            if checkbox is None or role_combo is None or not checkbox.isChecked():
                continue
            role = role_combo.currentText()
            if role == ROLE_IGNORE:
                continue
            item = self.table.item(row, 2)
            path = Path(item.data(_user_role()) or item.toolTip() or item.text())
            subject = self.table.item(row, 3).text().strip()
            session = self.table.item(row, 4).text().strip()
            acq = self.table.item(row, 5).text().strip()
            items.append(
                create_detection_for_role(
                    path,
                    role,
                    subject=subject,
                    session=session,
                    acq=acq,
                    cat12_format=self.cat12_format,
                    cat12_pattern=self.cat12_pattern,
                )
            )
        return items


class MigrationWorker(QObject):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        rows,
        bids_root,
        input_subdir="",
        overwrite=False,
        run_skullstrip=True,
        hd_bet_bin="hd-bet",
        device="cuda",
        disable_tta=False,
        subject_prefix="",
        cat12_format="auto",
        cat12_pattern="",
        include_metabolites=None,
    ):
        super().__init__()
        self.rows = rows
        self.bids_root = Path(bids_root)
        self.input_subdir = input_subdir or None
        self.overwrite = overwrite
        self.run_skullstrip = run_skullstrip
        self.hd_bet_bin = hd_bet_bin
        self.device = device
        self.disable_tta = disable_tta
        self.subject_prefix = str(subject_prefix or "").strip()
        self.cat12_format = cat12_format
        self.cat12_pattern = cat12_pattern or None
        self.include_metabolites = include_metabolites

    def _log(self, message):
        self.log.emit(str(message))

    def run(self):
        try:
            self.bids_root.mkdir(parents=True, exist_ok=True)
            active_rows = [row for row in self.rows if row["role"] != ROLE_IGNORE]
            t1_rows = [row for row in active_rows if row["role"] == ROLE_T1]
            total = len(active_rows) + (len(t1_rows) if self.run_skullstrip else 0)
            current = 0
            summary = {"mrsi": 0, "t1": 0, "cat12": 0, "skullstrip": 0, "errors": 0}
            skullstrip_inputs = []

            for row in active_rows:
                role = row["role"]
                source = Path(row["source_path"])
                subject = row["subject"].strip()
                session = normalize_session(row["session"])
                acq = row["acq"].strip()
                met = row.get("met", "").strip()
                desc = row.get("desc", "").strip()
                output_subject = apply_subject_prefix(subject, self.subject_prefix)
                self._log(f"[start] {role}: {source}")

                try:
                    if role == ROLE_MRSI:
                        if source.is_file():
                            if not output_subject or not session:
                                raise ValueError("MRSI row is missing subject or session.")
                            _, mrsi_status = migrate_mrsi_file(
                                str(source),
                                str(self.bids_root),
                                output_subject,
                                session,
                                overwrite=self.overwrite,
                                log_callback=self._log,
                                met_override=met or None,
                                desc_override=desc or None,
                            )
                            if mrsi_status != "skipped":
                                summary["mrsi"] += 1
                        else:
                            summary_info = migrate_session_folder(
                                str(source),
                                str(self.bids_root),
                                input_subdir=self.input_subdir,
                                migrate_anat=False,
                                overwrite=self.overwrite,
                                log_callback=self._log,
                                subject_override=output_subject or None,
                                session_override=session or None,
                                include_metabolites=self.include_metabolites,
                            )
                            summary["mrsi"] += len(summary_info["mrsi_files"])
                            for message in summary_info["skipped"] + summary_info["errors"]:
                                self._log(message)
                            summary["errors"] += len(summary_info["errors"])
                    elif role == ROLE_T1:
                        if not output_subject or not session:
                            raise ValueError("T1 row is missing subject or session.")
                        if not acq:
                            acq = _infer_anat_acq(source.name)
                        anat_out, anat_status = migrate_anatomical_file(
                            str(source),
                            str(self.bids_root),
                            output_subject,
                            session,
                            overwrite=self.overwrite,
                            log_callback=self._log,
                            acq_override=acq,
                        )
                        if anat_status != "skipped":
                            summary["t1"] += 1
                        skullstrip_inputs.append(Path(anat_out))
                    elif role == ROLE_CAT12:
                        match_info = dict(row.get("cat12_match") or {})
                        if not match_info:
                            detected = detect_cat12_file(
                                str(source),
                                input_format=self.cat12_format,
                                input_pattern=self.cat12_pattern,
                            )
                            if not detected:
                                raise ValueError("Could not match CAT12 filename.")
                            match_info = detected
                        if output_subject:
                            match_info["sub"] = output_subject
                        if session:
                            match_info["ses"] = session
                        if not match_info.get("sub") or not match_info.get("ses"):
                            raise ValueError("CAT12 row is missing subject or session.")
                        _, cat12_status = migrate_cat12_file(
                            match_info,
                            str(self.bids_root),
                            prefix=None,
                            overwrite=self.overwrite,
                            log_callback=self._log,
                        )
                        if cat12_status != "skipped":
                            summary["cat12"] += 1
                except Exception as exc:
                    summary["errors"] += 1
                    self._log(f"[error] {source}: {exc}")

                current += 1
                self.progress.emit(current, max(total, 1), f"Processed {role}")

            if self.run_skullstrip and skullstrip_inputs:
                for anat_path in skullstrip_inputs:
                    try:
                        self._log(f"[start] HD-BET: {anat_path}")
                        process_t1_files(
                            self.bids_root,
                            [anat_path],
                            hd_bet_bin=self.hd_bet_bin,
                            device=self.device,
                            disable_tta=self.disable_tta,
                            verbose=False,
                            overwrite=self.overwrite,
                            log_callback=self._log,
                        )
                        summary["skullstrip"] += 1
                    except Exception as exc:
                        summary["errors"] += 1
                        self._log(f"[error] HD-BET for {anat_path}: {exc}")
                    current += 1
                    self.progress.emit(current, max(total, 1), "Running HD-BET")

            self.finished.emit(summary)
        except Exception:
            self.failed.emit(traceback.format_exc())


class BidsMigrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._theme_name = "Light"
        self._compact_ui = False
        self._loading_table = False
        self._worker_thread = None
        self._worker = None
        self._detections = []

        self.setWindowTitle("Donald Migration")
        self.resize(1480, 880)
        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_theme(self._theme_name)
        self.statusBar().showMessage("Drop source folders or NIfTI files to begin.")

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        self.main_splitter = QSplitter(Qt.Horizontal if QT_LIB == 6 else Qt.Horizontal)
        self.main_splitter.setHandleWidth(10)
        main_layout.addWidget(self.main_splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        sources_group = QGroupBox("Sources")
        sources_layout = QVBoxLayout(sources_group)
        self.source_list = QListWidget()
        if QT_LIB == 6:
            self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.source_list.setSelectionMode(QAbstractItemView.SingleSelection)
        sources_layout.addWidget(self.source_list)

        source_buttons = QHBoxLayout()
        self.add_folders_button = QPushButton("Add Folders")
        self.add_folders_button.clicked.connect(self._add_folders)
        source_buttons.addWidget(self.add_folders_button)
        self.add_files_button = QPushButton("Add Files")
        self.add_files_button.clicked.connect(self._add_files)
        source_buttons.addWidget(self.add_files_button)
        sources_layout.addLayout(source_buttons)

        source_buttons_2 = QHBoxLayout()
        self.remove_source_button = QPushButton("Remove")
        self.remove_source_button.clicked.connect(self._remove_selected_source)
        source_buttons_2.addWidget(self.remove_source_button)
        self.clear_sources_button = QPushButton("Clear")
        self.clear_sources_button.clicked.connect(self._clear_sources)
        source_buttons_2.addWidget(self.clear_sources_button)
        self.scan_button = QPushButton("Scan Inputs")
        self.scan_button.clicked.connect(self.scan_sources)
        source_buttons_2.addWidget(self.scan_button)
        self.manual_select_button = QPushButton("Manual Select")
        self.manual_select_button.clicked.connect(self._open_manual_selector)
        source_buttons_2.addWidget(self.manual_select_button)
        sources_layout.addLayout(source_buttons_2)

        sources_hint = QLabel("Drag folders or NIfTI files here. The app will guess MRSI sessions, T1 images, and CAT12 maps.")
        sources_hint.setWordWrap(True)
        sources_layout.addWidget(sources_hint)

        destination_group = QGroupBox("Destination")
        destination_layout = QGridLayout(destination_group)
        destination_layout.addWidget(QLabel("BIDS root:"), 0, 0)
        self.destination_edit = QLineEdit("")
        self.destination_edit.setPlaceholderText("/path/to/bids_root")
        self.destination_edit.textChanged.connect(self._refresh_all_targets)
        destination_layout.addWidget(self.destination_edit, 0, 1)
        self.destination_button = QPushButton("Browse")
        self.destination_button.clicked.connect(self._browse_destination)
        destination_layout.addWidget(self.destination_button, 0, 2)

        options_group = QGroupBox("Options")
        options_layout = QGridLayout(options_group)
        options_layout.addWidget(QLabel("MRSI subdir:"), 0, 0)
        self.mrsi_subdir_edit = QLineEdit("")
        self.mrsi_subdir_edit.setPlaceholderText("Optional subfolder under each session")
        self.mrsi_subdir_edit.editingFinished.connect(self.scan_sources)
        options_layout.addWidget(self.mrsi_subdir_edit, 0, 1, 1, 2)

        options_layout.addWidget(QLabel("CAT12 format:"), 1, 0)
        self.cat12_format_combo = QComboBox()
        self.cat12_format_combo.addItems(["auto", "legacy", "simple", "bids"])
        self.cat12_format_combo.currentTextChanged.connect(self.scan_sources)
        options_layout.addWidget(self.cat12_format_combo, 1, 1, 1, 2)

        options_layout.addWidget(QLabel("CAT12 regex:"), 2, 0)
        self.cat12_pattern_edit = QLineEdit("")
        self.cat12_pattern_edit.setPlaceholderText("Optional custom regex")
        self.cat12_pattern_edit.editingFinished.connect(self.scan_sources)
        options_layout.addWidget(self.cat12_pattern_edit, 2, 1, 1, 2)

        options_layout.addWidget(QLabel("Subject prefix:"), 3, 0)
        self.subject_prefix_edit = QLineEdit("")
        self.subject_prefix_edit.setPlaceholderText("Prepended to every subject, e.g. CHUV")
        self.subject_prefix_edit.textChanged.connect(self._refresh_all_targets)
        options_layout.addWidget(self.subject_prefix_edit, 3, 1, 1, 2)

        options_layout.addWidget(QLabel("Ignore paths:"), 4, 0)
        self.ignore_terms_edit = QLineEdit(", ".join(DEFAULT_IGNORE_PATH_SUBSTRINGS))
        self.ignore_terms_edit.setPlaceholderText("Comma-separated substrings to skip")
        self.ignore_terms_edit.editingFinished.connect(self.scan_sources)
        options_layout.addWidget(self.ignore_terms_edit, 4, 1, 1, 2)

        options_layout.addWidget(QLabel("Select Metabolites:"), 5, 0)
        self.metabolite_mode_combo = QComboBox()
        self.metabolite_mode_combo.addItems(["default", "7T", "custom"])
        self.metabolite_mode_combo.currentTextChanged.connect(self._on_metabolite_mode_changed)
        options_layout.addWidget(self.metabolite_mode_combo, 5, 1, 1, 2)

        options_layout.addWidget(QLabel("Metabolites:"), 6, 0)
        self.metabolites_edit = QLineEdit(", ".join(DEFAULT_MRSI_METABOLITES))
        self.metabolites_edit.setPlaceholderText("Comma-separated metabolites")
        self.metabolites_edit.setEnabled(False)
        self.metabolites_edit.editingFinished.connect(self.scan_sources)
        options_layout.addWidget(self.metabolites_edit, 6, 1, 1, 2)

        options_layout.addWidget(QLabel("HD-BET bin:"), 7, 0)
        self.hd_bet_bin_edit = QLineEdit("hd-bet")
        options_layout.addWidget(self.hd_bet_bin_edit, 7, 1, 1, 2)

        options_layout.addWidget(QLabel("Device:"), 8, 0)
        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItems(["cuda", "cpu"])
        options_layout.addWidget(self.device_combo, 8, 1, 1, 2)

        self.overwrite_check = QCheckBox("Overwrite outputs")
        options_layout.addWidget(self.overwrite_check, 9, 0, 1, 3)
        self.skullstrip_check = QCheckBox("Run HD-BET after T1 migration")
        self.skullstrip_check.setChecked(True)
        options_layout.addWidget(self.skullstrip_check, 10, 0, 1, 3)
        self.disable_tta_check = QCheckBox("Disable TTA")
        options_layout.addWidget(self.disable_tta_check, 11, 0, 1, 3)

        left_layout.addWidget(sources_group)
        left_layout.addWidget(destination_group)
        left_layout.addWidget(options_group)
        left_layout.addStretch(1)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self.title_label = QLabel("Migration Preview")
        toolbar.addWidget(self.title_label, 1)
        toolbar.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Teya", "Donald"])
        self.theme_combo.setCurrentText(self._theme_name)
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        toolbar.addWidget(self.theme_combo)
        self.rescan_button = QPushButton("Refresh Preview")
        self.rescan_button.clicked.connect(self.scan_sources)
        toolbar.addWidget(self.rescan_button)
        self.run_button = QPushButton("Run Migration")
        self.run_button.clicked.connect(self._start_migration)
        toolbar.addWidget(self.run_button)
        center_layout.addLayout(toolbar)

        preview_group = QGroupBox("Detected Inputs")
        preview_layout = QVBoxLayout(preview_group)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["Role", "Source", "Subject", "Session", "Acq", "Met", "Desc", "Notes", "Exists?", "Target"]
        )
        header = self.table.horizontalHeader()
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._on_preview_header_clicked)
        if QT_LIB == 6:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        else:
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(7, QHeaderView.Stretch)
            header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.cellClicked.connect(self._on_preview_cell_clicked)
        preview_layout.addWidget(self.table)

        preview_hint = QLabel("You can change the guessed role and edit subject/session/acq values before running.")
        preview_hint.setWordWrap(True)
        preview_layout.addWidget(preview_hint)
        center_layout.addWidget(preview_group, 1)

        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Idle")
        progress_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, 1)
        center_layout.addLayout(progress_layout)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_sources_label = QLabel("Sources: 0")
        self.summary_detected_label = QLabel("Detected rows: 0")
        self.summary_mrsi_label = QLabel("MRSI: 0")
        self.summary_t1_label = QLabel("T1: 0")
        self.summary_cat12_label = QLabel("CAT12: 0")
        for label in (
            self.summary_sources_label,
            self.summary_detected_label,
            self.summary_mrsi_label,
            self.summary_t1_label,
            self.summary_cat12_label,
        ):
            summary_layout.addWidget(label)

        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.log_edit.clear)
        log_layout.addWidget(self.clear_log_button)

        right_layout.addWidget(summary_group)
        right_layout.addWidget(log_group, 1)

        self._styled_groups = [sources_group, destination_group, options_group, preview_group, summary_group, log_group]
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(center_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([370, 860, 360])

        self.setCentralWidget(central)
        self._update_summary()

    def _base_theme_stylesheet(self) -> str:
        if self._theme_name == "Dark":
            return (
                "QMainWindow, QWidget { background-color: #1f2430; color: #e5e7eb; }"
                "QLabel { color: #e5e7eb; }"
                "QLineEdit, QComboBox, QListWidget, QTableWidget, QProgressBar, QPlainTextEdit {"
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #2d3646; color: #e5e7eb; border: 1px solid #5f6d82; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #374256; }"
                "QPushButton:pressed { background: #2b3444; }"
                "QPushButton:disabled { color: #8e98a8; background: #252c38; border-color: #464f5e; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
                "QComboBox QAbstractItemView { background: #2a3140; color: #e5e7eb; selection-background-color: #3b82f6; }"
                "QStatusBar { background: #242a35; color: #e5e7eb; border-top: 1px solid #3d4556; }"
            )
        if self._theme_name == "Teya":
            return (
                "QMainWindow, QWidget { background-color: #ffd0e5; color: #0b7f7a; }"
                "QLabel { color: #0b7f7a; }"
                "QLineEdit, QComboBox, QListWidget, QTableWidget, QProgressBar, QPlainTextEdit {"
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #ffc0dc; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #ffb1d5; }"
                "QPushButton:pressed { background: #ffa3cd; }"
                "QPushButton:disabled { color: #68a9a6; background: #ffd9ea; border-color: #87cfcb; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
                "QComboBox QAbstractItemView { background: #ffe6f1; color: #0b7f7a; selection-background-color: #2ecfc9; }"
                "QStatusBar { background: #ffbddb; color: #0b7f7a; border-top: 1px solid #1db8b2; }"
            )
        if self._theme_name == "Donald":
            return (
                "QMainWindow, QWidget { background-color: #d97706; color: #ffffff; }"
                "QLabel { color: #ffffff; }"
                "QLineEdit, QComboBox, QListWidget, QTableWidget, QProgressBar, QPlainTextEdit {"
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #b85f00; color: #ffffff; border: 1px solid #f3a451; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #c76b06; }"
                "QPushButton:pressed { background: #a85400; }"
                "QPushButton:disabled { color: #ffe3be; background: #d58933; border-color: #f0b97c; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
                "QComboBox QAbstractItemView { background: #c96a04; color: #ffffff; selection-background-color: #2563eb; }"
                "QStatusBar { background: #b85f00; color: #ffffff; border-top: 1px solid #f3a451; }"
            )
        return (
            "QMainWindow, QWidget { background-color: #f3f5f8; color: #1f2937; }"
            "QLabel { color: #1f2937; }"
            "QLineEdit, QComboBox, QListWidget, QTableWidget, QProgressBar, QPlainTextEdit {"
            "background: #ffffff; color: #111827; border: 1px solid #c9d0da; border-radius: 5px; padding: 4px;"
            "}"
            "QPushButton {"
            "background: #ffffff; color: #1f2937; border: 1px solid #b7c0cc; border-radius: 6px; padding: 5px 10px;"
            "}"
            "QPushButton:hover { background: #edf2f7; }"
            "QPushButton:pressed { background: #e6ebf2; }"
            "QPushButton:disabled { color: #9099a5; background: #edf1f5; border-color: #d2d9e2; }"
            "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            "QComboBox QAbstractItemView { background: #ffffff; color: #111827; selection-background-color: #2563eb; }"
            "QStatusBar { background: #e8edf3; color: #1f2937; border-top: 1px solid #cfd7e2; }"
        )

    def _groupbox_stylesheet(self) -> str:
        if self._theme_name == "Dark":
            border_color, bg_color, text_color = "#4f5a6b", "#242b36", "#e5e7eb"
        elif self._theme_name == "Teya":
            border_color, bg_color, text_color = "#1db8b2", "#ffe0ef", "#0b7f7a"
        elif self._theme_name == "Donald":
            border_color, bg_color, text_color = "#f3a451", "#c96a04", "#ffffff"
        else:
            border_color, bg_color, text_color = "#c9ced6", "#fcfcfc", "#1f2937"
        return (
            "QGroupBox {"
            "font-weight: 600;"
            "font-size: 11pt;"
            f"color: {text_color};"
            f"border: 1px solid {border_color};"
            "border-radius: 6px;"
            "margin-top: 10px;"
            "padding-top: 6px;"
            f"background: {bg_color};"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "left: 10px;"
            "padding: 0 4px;"
            "}"
        )

    def _apply_theme(self, theme_name: str):
        theme = (theme_name or "Light").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Light"
        self._theme_name = theme
        self.setStyleSheet(self._base_theme_stylesheet())
        group_style = self._groupbox_stylesheet()
        for group in self._styled_groups:
            group.setStyleSheet(group_style)

    def _add_folders(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.Directory if QT_LIB == 6 else QFileDialog.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly if QT_LIB == 6 else QFileDialog.ShowDirsOnly, True)
        accepted = dialog.exec() if QT_LIB == 6 else dialog.exec_()
        if accepted:
            self._append_sources(dialog.selectedFiles())

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add NIfTI Files",
            "",
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if files:
            self._append_sources(files)

    def _append_sources(self, paths):
        existing = {self.source_list.item(index).text() for index in range(self.source_list.count())}
        for raw_path in paths:
            path = str(Path(raw_path).expanduser().resolve())
            if path in existing:
                continue
            self.source_list.addItem(QListWidgetItem(path))
            existing.add(path)
        self._update_summary()
        self.scan_sources()

    def _remove_selected_source(self):
        row = self.source_list.currentRow()
        if row >= 0:
            self.source_list.takeItem(row)
            self._update_summary()
            self.scan_sources()

    def _clear_sources(self):
        self.source_list.clear()
        self._detections = []
        self.table.setRowCount(0)
        self._update_summary()

    def _browse_destination(self):
        path = QFileDialog.getExistingDirectory(self, "Choose BIDS Root")
        if path:
            self.destination_edit.setText(path)

    def _source_paths(self):
        return [self.source_list.item(index).text() for index in range(self.source_list.count())]

    def _ignore_terms(self):
        return parse_ignore_terms(self.ignore_terms_edit.text())

    def _mrsi_include_metabolites(self):
        mode = self.metabolite_mode_combo.currentText()
        if mode == "7T":
            return list(MRSI_7T_METABOLITES)
        if mode != "custom":
            return list(DEFAULT_MRSI_METABOLITES)
        custom_metabolites = parse_csv_terms(self.metabolites_edit.text())
        return custom_metabolites if custom_metabolites else list(DEFAULT_MRSI_METABOLITES)

    def _on_metabolite_mode_changed(self, mode):
        custom = mode == "custom"
        self.metabolites_edit.setEnabled(custom)
        if mode == "7T":
            self.metabolites_edit.setText(", ".join(MRSI_7T_METABOLITES))
        elif not custom:
            self.metabolites_edit.setText(", ".join(DEFAULT_MRSI_METABOLITES))
        self.scan_sources()

    def _merge_detections(self, items, replace_existing=False):
        merged = {}
        if not replace_existing:
            for detection in self._detections:
                merged[os.path.realpath(detection.source_path)] = detection
        for detection in items:
            merged[os.path.realpath(detection.source_path)] = detection
        self._detections = sorted(
            merged.values(),
            key=lambda item: (item.detected_kind, item.subject, item.session, item.source_path),
        )
        self._populate_table()

    def _open_manual_selector(self, replace_existing=False):
        source_paths = self._source_paths()
        if not source_paths:
            QMessageBox.information(self, "No Sources", "Add at least one folder or file first.")
            return
        dialog = ManualSelectionDialog(
            source_paths,
            cat12_format=self.cat12_format_combo.currentText(),
            cat12_pattern=self.cat12_pattern_edit.text().strip() or None,
            ignore_terms=self._ignore_terms(),
            include_metabolites=self._mrsi_include_metabolites(),
            parent=self,
        )
        dialog.setStyleSheet(self.styleSheet())
        accepted = dialog.exec() if QT_LIB == 6 else dialog.exec_()
        if not accepted:
            return
        selected_items = dialog.selected_items()
        if not selected_items:
            self._append_log("[manual] no files selected")
            return
        self._append_log(f"[manual] added {len(selected_items)} manually selected files")
        self._merge_detections(selected_items, replace_existing=replace_existing)
        self.statusBar().showMessage(f"Added {len(selected_items)} manually selected files.")

    def scan_sources(self):
        source_paths = self._source_paths()
        self._detections = scan_input_paths(
            source_paths,
            input_subdir=self.mrsi_subdir_edit.text().strip() or None,
            cat12_format=self.cat12_format_combo.currentText(),
            cat12_pattern=self.cat12_pattern_edit.text().strip() or None,
            ignore_terms=self._ignore_terms(),
            include_metabolites=self._mrsi_include_metabolites(),
        )
        self._populate_table()
        self._append_log(f"[scan] found {len(self._detections)} candidate items")
        if source_paths and not self._detections:
            self._append_log("[scan] no candidates found automatically")
            choice = QMessageBox.question(
                self,
                "No Candidates Found",
                "No candidates were found automatically. Do you want to open the manual file selector?",
            )
            if choice == QMessageBox.Yes:
                self._open_manual_selector(replace_existing=True)
                return
        self.statusBar().showMessage(f"Scanned {len(source_paths)} sources.")

    def _populate_table(self):
        self._loading_table = True
        self.table.setRowCount(len(self._detections))
        for row, detection in enumerate(self._detections):
            role_combo = QComboBox()
            role_combo.addItems(ROLE_OPTIONS)
            role_combo.setCurrentText(detection.default_role)
            role_combo.currentTextChanged.connect(lambda _value, r=row: self._on_role_changed(r))
            self.table.setCellWidget(row, 0, role_combo)

            source_item = QTableWidgetItem(_path_display_name(detection.source_path))
            source_item.setToolTip(detection.source_path)
            source_item.setData(_user_role(), detection.source_path)
            _set_noneditable(source_item)
            self.table.setItem(row, 1, source_item)

            self.table.setItem(row, 2, QTableWidgetItem(detection.subject))
            self.table.setItem(row, 3, QTableWidgetItem(detection.session))
            self.table.setItem(row, 4, QTableWidgetItem(detection.acq))
            self.table.setItem(row, 5, QTableWidgetItem(detection.met))
            self.table.setItem(row, 6, QTableWidgetItem(detection.desc))

            notes = detection.notes
            if detection.confidence:
                notes = f"{notes} [{detection.confidence}]".strip() if notes else f"[{detection.confidence}]"
            notes_item = QTableWidgetItem(notes)
            _set_noneditable(notes_item)
            self.table.setItem(row, 7, notes_item)

            exists_item = QTableWidgetItem("")
            _set_noneditable(exists_item)
            exists_item.setTextAlignment(_center_alignment())
            self.table.setItem(row, 8, exists_item)

            target_item = QTableWidgetItem("")
            _set_noneditable(target_item)
            self.table.setItem(row, 9, target_item)
            self._refresh_row_target(row)
        self._loading_table = False
        self._update_summary()

    def _role_for_row(self, row):
        combo = self.table.cellWidget(row, 0)
        return combo.currentText() if combo is not None else ROLE_IGNORE

    def _text_for_row(self, row, column):
        item = self.table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _refresh_row_target(self, row):
        if row < 0 or row >= len(self._detections):
            return
        exists_item = self.table.item(row, 8)
        target_item = self.table.item(row, 9)
        if exists_item is None or target_item is None:
            return
        detection = self._detections[row]
        role = self._role_for_row(row)
        subject = self._text_for_row(row, 2) or detection.subject
        session = normalize_session(self._text_for_row(row, 3) or detection.session)
        acq = self._text_for_row(row, 4) or detection.acq or _infer_anat_acq(Path(detection.source_path).name)
        met = self._text_for_row(row, 5)
        desc = self._text_for_row(row, 6)
        output_subject = apply_subject_prefix(subject, self.subject_prefix_edit.text().strip())
        root = self.destination_edit.text().strip() or "<BIDS_ROOT>"

        if role == ROLE_MRSI:
            if not output_subject or not session:
                target = "Missing subject/session"
                target_is_path = False
            elif Path(detection.source_path).is_file():
                try:
                    target = build_mrsi_destination(
                        detection.source_path,
                        root,
                        output_subject,
                        session,
                        met_override=met or None,
                        desc_override=desc or None,
                    )
                    target_is_path = True
                except Exception:
                    target = "Unrecognized MRSI filename"
                    target_is_path = False
            else:
                target = os.path.join(root, "derivatives", "mrsi-orig", f"sub-{output_subject}", f"ses-{session}")
                target_is_path = True
        elif role == ROLE_T1:
            if not output_subject or not session:
                target = "Missing subject/session"
                target_is_path = False
            else:
                target = os.path.join(
                    root,
                    f"sub-{output_subject}",
                    f"ses-{session}",
                    "anat",
                    f"sub-{output_subject}_ses-{session}_acq-{acq}_T1w.nii.gz",
                )
                target_is_path = True
        elif role == ROLE_CAT12:
            if not output_subject or not session:
                target = "Missing subject/session"
                target_is_path = False
            else:
                match_info = dict(detection.cat12_match or {})
                if match_info:
                    if output_subject:
                        match_info["sub"] = output_subject
                    if session:
                        match_info["ses"] = session
                    target = build_cat12_destination(
                        match_info,
                        root,
                        prefix=None,
                    )
                    target_is_path = True
                else:
                    target = "Pattern mismatch"
                    target_is_path = False
        else:
            target = ""
            target_is_path = False
        _set_exists_indicator(exists_item, target, target_is_path)
        _set_path_item_display(target_item, target, target_is_path)

    def _refresh_all_targets(self):
        if self._loading_table:
            return
        for row in range(len(self._detections)):
            self._refresh_row_target(row)
        self._update_summary()

    def _sync_detection_from_row(self, row):
        if row < 0 or row >= len(self._detections):
            return
        detection = self._detections[row]
        detection.default_role = self._role_for_row(row)
        detection.subject = self._text_for_row(row, 2)
        detection.session = normalize_session(self._text_for_row(row, 3))
        detection.acq = self._text_for_row(row, 4)
        detection.met = self._text_for_row(row, 5)
        detection.desc = self._text_for_row(row, 6)

    def _exists_rank_for_row(self, row):
        item = self.table.item(row, 8)
        if item is None:
            return 2
        rank = item.data(_user_role())
        try:
            return int(rank)
        except (TypeError, ValueError):
            return 2

    def _sort_by_exists(self):
        if self._loading_table:
            return
        self._refresh_all_targets()
        ranked_detections = []
        for row, detection in enumerate(self._detections):
            self._sync_detection_from_row(row)
            ranked_detections.append(
                (
                    self._exists_rank_for_row(row),
                    detection.subject,
                    detection.session,
                    detection.source_path,
                    detection,
                )
            )
        ranked_detections.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        self._detections = [item[-1] for item in ranked_detections]
        self._populate_table()
        self.statusBar().showMessage("Sorted by Exists?: missing outputs first.")

    def _on_preview_header_clicked(self, section):
        if section == 8:
            self._sort_by_exists()

    def _on_preview_cell_clicked(self, row, column):
        if column == 8:
            self._sort_by_exists()

    def _on_table_item_changed(self, item):
        if self._loading_table or item is None:
            return
        row = item.row()
        if item.column() == 3:
            normalized = normalize_session(item.text())
            if normalized and normalized != item.text():
                self._loading_table = True
                item.setText(normalized)
                self._loading_table = False
        self._refresh_row_target(row)

    def _on_role_changed(self, row):
        self._refresh_row_target(row)
        self._update_summary()

    def _update_summary(self):
        self.summary_sources_label.setText(f"Sources: {self.source_list.count()}")
        self.summary_detected_label.setText(f"Detected rows: {len(self._detections)}")
        mrsi_count = 0
        t1_count = 0
        cat12_count = 0
        for row in range(len(self._detections)):
            role = self._role_for_row(row)
            if role == ROLE_MRSI:
                mrsi_count += 1
            elif role == ROLE_T1:
                t1_count += 1
            elif role == ROLE_CAT12:
                cat12_count += 1
        self.summary_mrsi_label.setText(f"MRSI: {mrsi_count}")
        self.summary_t1_label.setText(f"T1: {t1_count}")
        self.summary_cat12_label.setText(f"CAT12: {cat12_count}")

    def _append_log(self, message):
        self.log_edit.appendPlainText(str(message))

    def _collect_rows(self):
        rows = []
        for index, detection in enumerate(self._detections):
            rows.append(
                {
                    "role": self._role_for_row(index),
                    "source_path": detection.source_path,
                    "subject": self._text_for_row(index, 2),
                    "session": self._text_for_row(index, 3),
                    "acq": self._text_for_row(index, 4),
                    "met": self._text_for_row(index, 5),
                    "desc": self._text_for_row(index, 6),
                    "cat12_match": detection.cat12_match,
                }
            )
        return rows

    def _set_running(self, running: bool):
        self.run_button.setEnabled(not running)
        self.scan_button.setEnabled(not running)
        self.rescan_button.setEnabled(not running)
        self.add_files_button.setEnabled(not running)
        self.add_folders_button.setEnabled(not running)
        self.remove_source_button.setEnabled(not running)
        self.clear_sources_button.setEnabled(not running)
        self.manual_select_button.setEnabled(not running)

    def _start_migration(self):
        bids_root = self.destination_edit.text().strip()
        if not bids_root:
            QMessageBox.warning(self, "Destination Required", "Choose a BIDS destination first.")
            return
        rows = self._collect_rows()
        if not any(row["role"] != ROLE_IGNORE for row in rows):
            QMessageBox.information(self, "Nothing To Run", "There are no active rows to migrate.")
            return

        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting...")
        self._append_log("[run] migration started")
        self._set_running(True)

        self._worker_thread = QThread(self)
        self._worker = MigrationWorker(
            rows=rows,
            bids_root=bids_root,
            input_subdir=self.mrsi_subdir_edit.text().strip(),
            overwrite=self.overwrite_check.isChecked(),
            run_skullstrip=self.skullstrip_check.isChecked(),
            hd_bet_bin=self.hd_bet_bin_edit.text().strip() or "hd-bet",
            device=self.device_combo.currentText().strip() or "cuda",
            disable_tta=self.disable_tta_check.isChecked(),
            subject_prefix=self.subject_prefix_edit.text().strip(),
            cat12_format=self.cat12_format_combo.currentText(),
            cat12_pattern=self.cat12_pattern_edit.text().strip(),
            include_metabolites=self._mrsi_include_metabolites(),
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _on_worker_progress(self, current, total, message):
        total = max(total, 1)
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"{message} ({current}/{total})")
        self.statusBar().showMessage(self.progress_label.text())

    def _on_worker_finished(self, summary):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Completed")
        self.statusBar().showMessage("Migration completed.")
        self._append_log(
            f"[done] mrsi={summary['mrsi']} t1={summary['t1']} cat12={summary['cat12']} skullstrip={summary['skullstrip']} errors={summary['errors']}"
        )
        QMessageBox.information(
            self,
            "Migration Complete",
            (
                f"MRSI outputs: {summary['mrsi']}\n"
                f"T1 migrations: {summary['t1']}\n"
                f"CAT12 files: {summary['cat12']}\n"
                f"HD-BET runs: {summary['skullstrip']}\n"
                f"Errors: {summary['errors']}"
            ),
        )

    def _on_worker_failed(self, stack_trace):
        self._set_running(False)
        self.progress_label.setText("Failed")
        self.statusBar().showMessage("Migration failed.")
        self._append_log(stack_trace)
        QMessageBox.critical(self, "Migration Failed", stack_trace)

    def _cleanup_worker(self):
        if self._worker is not None:
            self._worker.deleteLater()
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        self._worker = None
        self._worker_thread = None

    def dragEnterEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            local_path = Path(url.toLocalFile())
            if local_path.is_dir() or is_nifti_path(local_path):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            local_path = Path(url.toLocalFile())
            if local_path.is_dir() or is_nifti_path(local_path):
                paths.append(str(local_path))
        if not paths:
            event.ignore()
            return
        self._append_sources(paths)
        event.acceptProposedAction()


def main():
    app = QApplication(sys.argv)
    if sys.platform.startswith("linux"):
        app.setApplicationName("donald-migration")
    window = BidsMigrationWindow()
    window.show()
    return app.exec() if QT_LIB == 6 else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
