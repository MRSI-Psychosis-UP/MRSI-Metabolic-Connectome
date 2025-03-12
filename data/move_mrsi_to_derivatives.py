#!/usr/bin/env python3

import os
import re
import shutil
import sys

def rearrange_mrsi_files(bids_root):
    """
    1) Recursively find all files containing '_mrsi' in their names.
    2) Detect 'space-XYZ' from the filename using a regex.
    3) Detect 'sub-???' and 'ses-???' from the directory path (or fallback to filename).
    4) Move each file to: derivatives/mrsi/space-XYZ/sub-???/ses-???/<filename>
    """

    derivatives_dir = os.path.join(bids_root, 'derivatives', 'mrsi')

    # Walk the entire BIDS directory structure (except any existing derivatives folder)
    for root, dirs, files in os.walk(bids_root):
        # Skip if we're already inside a derivatives folder
        if 'derivatives' in root.split(os.sep):
            continue

        for filename in files:
            # We're targeting files that have '_mrsi' in their name
            if '_mrsi' in filename:
                # Look for a pattern like '_space-<LABEL>_' in the filename
                space_match = re.search(r'_space-([^_]+)_', filename)
                if space_match:
                    space_label = space_match.group(1)
                else:
                    # If there's no space-XXX in the filename, skip it
                    # or handle differently if needed
                    continue

                # Figure out sub-??? and ses-??? from the path
                sub_label = None
                ses_label = None

                path_parts = root.split(os.sep)
                for part in path_parts:
                    if part.startswith('sub-'):
                        sub_label = part
                    elif part.startswith('ses-'):
                        ses_label = part

                # If we didn't find sub-/ses- in the folder structure, fallback to searching in the filename
                if not sub_label:
                    sub_match = re.search(r'(sub-[a-zA-Z0-9]+)', filename)
                    if sub_match:
                        sub_label = sub_match.group(1)

                if not ses_label:
                    ses_match = re.search(r'(ses-[a-zA-Z0-9]+)', filename)
                    if ses_match:
                        ses_label = ses_match.group(1)

                # Build the new directory
                # e.g., derivatives/mrsi/space-orig/sub-S001/ses-V1/
                new_root = os.path.join(derivatives_dir, f"space-{space_label}")
                if sub_label:
                    new_root = os.path.join(new_root, sub_label)
                if ses_label:
                    new_root = os.path.join(new_root, ses_label)

                os.makedirs(new_root, exist_ok=True)

                # Construct old and new full paths
                old_path = os.path.join(root, filename)
                new_path = os.path.join(new_root, filename)

                print(f"Moving:\n  {old_path}\n  => {new_path}\n")
                # shutil.move(old_path, new_path)
                shutil.copy(old_path, new_path)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/bids_dataset")
        sys.exit(1)

    bids_root = sys.argv[1]
    if not os.path.isdir(bids_root):
        print(f"Error: {bids_root} is not a valid directory.")
        sys.exit(1)

    rearrange_mrsi_files(bids_root)

if __name__ == "__main__":
    main()
