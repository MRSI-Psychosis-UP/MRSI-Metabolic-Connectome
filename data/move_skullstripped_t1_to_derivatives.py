#!/usr/bin/env python3

import os
import re
import shutil
import sys

def move_skullstripped_t1(bids_root):
    """
    1) Recursively find T1 files containing '_brain' or '_brainmask' in their name.
    2) Parse sub-XXX and ses-XXX from directory structure or filename.
    3) Rename them to a derivative-friendly naming scheme:
       - _T1w_brain -> _desc-brain_T1w
       - _T1w_brainmask -> _desc-brain_mask
    4) Move them to derivatives/skullstrip/sub-XXX/ses-XXX/anat/
    """

    # We'll place them in: <bids_root>/derivatives/skullstrip/<sub>/<ses>/anat/
    derivatives_root = os.path.join(bids_root, "derivatives", "skullstrip")

    # Regex to capture files with T1w + "brain" or "brainmask"
    # e.g., sub-S001_ses-V1_run-01_acq-memprage_T1w_brain.nii.gz
    #       sub-S001_ses-V1_run-01_acq-memprage_T1w_brainmask.nii.gz
    pattern_brain = re.compile(r"(.*)_T1w_brain(\..+)$")       # group(1) is prefix, group(2) is extension
    pattern_brainmask = re.compile(r"(.*)_T1w_brainmask(\..+)$")  # similarly

    for root, dirs, files in os.walk(bids_root):
        # Skip if we are already under derivatives
        if 'derivatives' in root.split(os.sep):
            continue

        for filename in files:
            old_path = os.path.join(root, filename)

            # Check for _T1w_brain
            match_brain = pattern_brain.match(filename)
            if match_brain:
                prefix = match_brain.group(1)  # everything before '_T1w_brain'
                ext = match_brain.group(2)     # file extension (e.g. ".nii.gz", ".json", etc.)
                # e.g. new filename: sub-S001_ses-V1_run-01_acq-memprage_desc-brain_T1w.nii.gz
                new_filename = f"{prefix}_desc-brain_T1w{ext}"
                move_file(old_path, new_filename, derivatives_root)

            # Check for _T1w_brainmask
            match_brainmask = pattern_brainmask.match(filename)
            if match_brainmask:
                prefix = match_brainmask.group(1)
                ext = match_brainmask.group(2)
                # e.g. new filename: sub-S001_ses-V1_run-01_acq-memprage_desc-brain_mask.nii.gz
                new_filename = f"{prefix}_desc-brain_mask{ext}"
                move_file(old_path, new_filename, derivatives_root)


def move_file(old_path, new_filename, derivatives_root):
    """
    Move file from old_path to derivatives/skullstrip/sub-XXX/ses-XXX/anat/new_filename
    sub-XXX / ses-XXX gleaned from either the path or the prefix.
    """
    # Extract sub-XXX and ses-XXX from old_path or from new_filename
    # path_parts = old_path.split(os.sep)
    path_parts = os.path.dirname(old_path).split(os.sep)

    sub_label, ses_label = None, None
    # 1) Try to find sub- and ses- in the path
    for p in path_parts:
        if p.startswith("sub-"):
            sub_label = p
        elif p.startswith("ses-"):
            ses_label = p

    # 2) If not found, try matching in the filename itself
    if not sub_label:
        sub_match = re.search(r"(sub-\w+)", new_filename)
        if sub_match:
            sub_label = sub_match.group(1)
    if not ses_label:
        ses_match = re.search(r"(ses-\w+)", new_filename)
        if ses_match:
            ses_label = ses_match.group(1)

    # Construct the new directory: derivatives/skullstrip[/sub-XX][/ses-XX]/anat
    new_dir = derivatives_root
    if sub_label:
        new_dir = os.path.join(new_dir, sub_label)
    if ses_label:
        new_dir = os.path.join(new_dir, ses_label)
    new_dir = os.path.join(new_dir)

    os.makedirs(new_dir, exist_ok=True)
    new_path = os.path.join(new_dir, new_filename)

    # Move if old_path still exists (it might have been moved already)
    if os.path.exists(old_path):
        print(f"Moving:\n  {old_path}\n  => {new_path}\n")
        shutil.move(old_path, new_path)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/bids_dataset")
        sys.exit(1)

    bids_root = sys.argv[1]
    if not os.path.isdir(bids_root):
        print(f"Error: {bids_root} is not a valid directory.")
        sys.exit(1)

    move_skullstripped_t1(bids_root)

if __name__ == "__main__":
    main()
