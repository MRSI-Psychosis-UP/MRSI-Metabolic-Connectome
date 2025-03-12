import os

def rename_spectroscopy_to_mrsi(bids_root):
    """
    Recursively rename 'spectroscopy' folders to 'mrsi'
    and any files containing '_spectroscopy' to '_mrsi' in the given BIDS root directory.
    """

    # Walk bottom-up so we rename files BEFORE renaming their parent folder.
    for root, dirs, files in os.walk(bids_root, topdown=False):

        # 1) Rename files containing '_spectroscopy' to '_mrsi'
        for filename in files:
            if '_spectroscopy' in filename:
                old_path = os.path.join(root, filename)
                # new_filename = filename.replace('_spectroscopy', '_mrsi')
                new_path = old_path.replace("_spectroscopy","_mrsi")
                
                print(f"Renaming file:\n  {old_path}\n  => {new_path}\n")
                # os.rename(old_path, new_path)
    for root, dirs, files in os.walk(bids_root, topdown=False):
        # 1) Rename files containing '_spectroscopy' to '_mrsi'
        for filename in files:
            if '/spectroscopy/' in filename:
                old_path = os.path.join(root, filename)
                # new_filename = filename.replace('_spectroscopy', '_mrsi')
                new_path = old_path.replace("/spectroscopy/","/mrsi/")
                print(f"Renaming file:\n  {old_path}\n  => {new_path}\n")
                # os.rename(old_path, new_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/bids_dataset")
        sys.exit(1)

    bids_root_dir = sys.argv[1]

    if not os.path.isdir(bids_root_dir):
        print(f"Error: {bids_root_dir} is not a valid directory.")
        sys.exit(1)

    rename_spectroscopy_to_mrsi(bids_root_dir)
