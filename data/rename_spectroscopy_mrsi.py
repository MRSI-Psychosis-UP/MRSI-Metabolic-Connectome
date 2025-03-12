import os
import sys

def rename_spectroscopy_to_mrsi(bids_root):
    """
    1) Recursively rename any files containing '_spectroscopy' to '_mrsi'.
    2) Then rename any directory literally named 'spectroscopy' to 'mrsi'.
    """

    # -- PASS 1: Rename files first --
    for root, dirs, files in os.walk(bids_root, topdown=False):
        for filename in files:
            if '_spectroscopy' in filename:
                old_path = os.path.join(root, filename)
                new_path = old_path.replace("_spectroscopy", "_mrsi")

                print(f"Renaming file:\n  {old_path}\n  => {new_path}\n")
                os.rename(old_path, new_path)

    # -- PASS 2: Rename folders named 'spectroscopy' to 'mrsi' --
    for root, dirs, files in os.walk(bids_root, topdown=False):
        base_name = os.path.basename(root)
        if base_name == "spectroscopy":
            # Build the new path by replacing 'spectroscopy' with 'mrsi'
            new_root = root.replace("spectroscopy", "mrsi")

            print(f"Renaming directory:\n  {root}\n  => {new_root}\n")
            os.rename(root, new_root)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/bids_dataset")
        sys.exit(1)

    bids_root_dir = sys.argv[1]
    if not os.path.isdir(bids_root_dir):
        print(f"Error: {bids_root_dir} is not a valid directory.")
        sys.exit(1)

    # Safety tip: consider making a backup of your dataset before running this script.
    rename_spectroscopy_to_mrsi(bids_root_dir)
