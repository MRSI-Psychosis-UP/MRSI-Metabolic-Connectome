import numpy as np
import matplotlib as plt
import math, sys, os, re
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv
from os.path import join, split
from pathlib import Path
from .debug import Debug


debug=Debug()

def find_root_path():
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != "Connectome":
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            return None
    return current_path

class DataUtils:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file

        self.ROOTPATH = find_root_path()

        package_root = Path(__file__).resolve().parents[3]
        env_dev_path = os.getenv("DEVANALYSEPATH")
        if env_dev_path is None or str(env_dev_path).strip() in {"", "."}:
            self.DEVANALYSEPATH = str(package_root)
            debug.warning("DEVANALYSEPATH env empty, set to", self.DEVANALYSEPATH)
        else:
            self.DEVANALYSEPATH = str(Path(env_dev_path).expanduser())

        self.ANARESULTSPATH = join(self.DEVANALYSEPATH, "results")
        self.DEVDATAPATH    = join(self.DEVANALYSEPATH, "data")
        self.ANALOGPATH     = join(self.DEVANALYSEPATH, "logs")
        self.BIDS_STRUCTURE_PATH = join(self.DEVANALYSEPATH, "data", "structure.json")

        # Create necessary directories if they don't exist
        os.makedirs(self.ANALOGPATH, exist_ok=True)
        os.makedirs(self.ANARESULTSPATH, exist_ok=True)

        # Load BIDSDATAPATH from .env
        if os.getenv("BIDSDATAPATH") is None or os.getenv("BIDSDATAPATH")==".":
            debug.warning("BIDSDATAPATH env empty, set to",join(self.DEVDATAPATH,"BIDS"))
            self.BIDSDATAPATH = join(self.DEVDATAPATH,"BIDS")
        else:
            self.BIDSDATAPATH = os.getenv("BIDSDATAPATH")


if __name__=='__main__':
    u = DataUtils()
    debug.info("DEVANALYSEPATH",u.DEVANALYSEPATH)
    debug.info("BIDSDATAPATH",u.BIDSDATAPATH)



