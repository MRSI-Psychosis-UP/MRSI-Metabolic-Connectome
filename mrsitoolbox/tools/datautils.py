import numpy as np
import matplotlib as plt
import math, sys, os, re
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv
from os.path import join, split
from mrsitoolbox.tools.debug import Debug


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

        # DEVANALYSEPATH is derived from the current script's location.
        self.DEVANALYSEPATH = os.getenv("DEVANALYSEPATH")
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





