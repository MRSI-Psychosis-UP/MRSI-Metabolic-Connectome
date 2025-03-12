#!/usr/bin/env python3
import os
import re
import sys

def fix_t1_naming(bids_root):
    """
    Recursively finds T1w files that have the wrong ordering of 'run' and 'acq'
    (i.e., '_run-XX_acq-YYY_T1w') and renames them to the correct BIDS order
    '_acq-YYY_run-XX_T1w'.
    """

    pattern = re.compile(r'(_run-\d+)_acq-([^_]+)_T1w(\..+)$')
    # Explanation of the regex groups:
    #   1) (_run-\d+)  => e.g. "_run-01" or "_run-2"
    #   2) ([^_]+)     => one or more non-underscore chars (the acq label)
    #   3) (\..+)$     => the file extension (e.g., ".nii.gz", ".json")

    for root, dirs, files in os.walk(bids_root):
        for filename in files:
            # Only look at files containing T1w in the name (e.g., .nii.gz or .json)
            if 'T1w' not in filename:
                continue

            # Check if it matches the pattern '_run-XX_acq-YYY_T1w'
            match = pattern.search(filename)
            if match:
                old_path = os.path.join(root, filename)
                # Create the new filename with correct ordering '_acq-YYY_run-XX_T1w'
                new_filename = pattern.sub(r'_acq-\2\1_T1w\3', filename)
                # Explanation of the replacement:
                #  r'_acq-\2\1_T1w\3'
                #    \2 => the captured acq label
                #    \1 => the entire run-XX part (including the underscore)
                #    \3 => the file extension (e.g., .nii.gz or .json)
                new_path = os.path.join(root, new_filename)
                print(f"Renaming:\n  {old_path}\n  => {new_path}\n")
                os.rename(old_path, new_path)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/bids_dataset")
        sys.exit(1)

    bids_root = sys.argv[1]
    if not os.path.isdir(bids_root):
        print(f"Error: {bids_root} is not a valid directory.")
        sys.exit(1)

    fix_t1_naming(bids_root)

if __name__ == "__main__":
    main()
