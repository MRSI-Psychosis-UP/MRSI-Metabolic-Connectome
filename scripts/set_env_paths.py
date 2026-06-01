import os
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv, set_key

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load existing .env file
env_path = os.path.join(REPO_ROOT, '.env')
load_dotenv(dotenv_path=env_path)

# Initialize Tkinter root and hide the main window
root = tk.Tk()
root.withdraw()

# Get the current BIDSDATAPATH from .env or set as empty string if not present
current_bidsdatapath = os.getenv("BIDSDATAPATH", "")

# Display current BIDSDATAPATH and ask user to select a new folder if desired
if current_bidsdatapath:
    print(f"Current BIDSDATAPATH is: {current_bidsdatapath}")
    use_current = input("Press ENTER to keep this path or type 'n' to select a new folder: ").lower() != 'n'
    if use_current:
        new_bidsdatapath = current_bidsdatapath
    else:
        new_bidsdatapath = filedialog.askdirectory(title="Select new BIDSDATAPATH folder")
else:
    print("No BIDSDATAPATH is currently set.")
    new_bidsdatapath = filedialog.askdirectory(title="Select BIDSDATAPATH folder")

# Set BIDSDATAPATH in .env file
set_key(env_path, "BIDSDATAPATH", new_bidsdatapath)
print(f"BIDSDATAPATH set to {new_bidsdatapath} in .env")

# Set DEVANALYSEPATH to the repository root
set_key(env_path, "DEVANALYSEPATH", REPO_ROOT)
print(f"DEVANALYSEPATH set to {REPO_ROOT} in .env")

# Confirm by displaying the final .env content
with open(env_path, 'r') as f:
    print("\nFinal contents of .env:")
    print(f.read())
