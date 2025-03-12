import os, sys
import numpy as np
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.datautils import DataUtils
from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
from os.path import join, split, exists
from tools.mridata import MRIData
import argparse
from nilearn import datasets  





dutils     = DataUtils()
debug      = Debug()
regtools   = RegTools()
reg        = Registration()



parser = argparse.ArgumentParser(description="Process some input parameters.")
# Parse arguments
parser.add_argument('--group', type=str, default="Dummy-Project") 
parser.add_argument('--nthreads',type=int,default=4,help="Number of CPU threads [default=4]")
parser.add_argument('--subject_id', type=str, help='subject id', default="S001")
parser.add_argument('--session', type=str, help='recording session',choices=['V1', 'V2', 'V3','V4','V5'], default="V1")
parser.add_argument('--overwrite',type=int,default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
parser.add_argument('--t1_pattern',type=str,default="_run-01_acq-memprage_T1w_brain",help="T1w file pattern e.g _run-01_acq-memprage_T1w_brain")


args           = parser.parse_args()
GROUP          = args.group
NTHREADS       = args.nthreads
subject_id     = args.subject_id
session        = args.session
overwrite_flag = bool(args.overwrite)
t1pattern      = args.t1_pattern

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(NTHREADS)

#
mridata            = MRIData(subject_id, session,GROUP,t1pattern)
ftools             = FileTools(GROUP)
BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")


################################################

debug.separator()
debug.title(f"Processing {subject_id}-{session} ")
############ T1w to MNI Registration ##################  
transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","anat")
transform_prefix          = f"sub-{subject_id}_ses-{session}_desc-t1w_to_mni"
transform_dir_prefix_path = join(transform_dir_path,f"{transform_prefix}")
if not exists(transform_dir_prefix_path) or overwrite_flag:
    debug.warning(f"{transform_prefix} to T1w Registration not found or not up to date")
    syn_tx,_          = reg.register(fixed_input  = datasets.load_mni152_template(),
                                    moving_input  = mridata.data["t1w"]["brain"]["orig"]["path"],
                                    fixed_mask    = None, 
                                    moving_mask   = None,
                                    transform     = "s",
                                    verbose       = 0)
    # Save Transform
    os.makedirs(transform_dir_path,exist_ok=True)
    regtools.save_all_transforms(syn_tx,transform_dir_prefix_path)

    #################################
debug.title("DONE")
debug.separator()











    






