import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
import random, argparse
from os.path import join, split
from connectomics.parcellate import Parcellate
from tools.mridata import MRIData
from filters.biharmonic import BiHarmonic

METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]

dutils   = DataUtils()
debug    = Debug()
regtools = RegTools()
reg      = Registration()
parc     = Parcellate()
bhfilt   = BiHarmonic()



parser = argparse.ArgumentParser(description="Process some input parameters.")
# Parse arguments
parser.add_argument('--group'     , type=str, default = "Dummy-Project") 
parser.add_argument('--nthreads'  , type=int, default = 4,help="Number of CPU threads [default=4]")
parser.add_argument('--ref_met'   , type=str, default = "CrPCr",help="Reference metabolite to be coregistered with T1 [CrPCr]")
parser.add_argument('--subject_id', type=str, help='subject id', default="S001")
parser.add_argument('--session'   , type=str, help='recording session',choices=['V1', 'V2', 'V3','V4','V5'], default="V1")
parser.add_argument('--overwrite' , type=int, default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
parser.add_argument('--t1_pattern', type=str, default="_run-01_acq-memprage_",help="T1w file pattern e.g _run-01_acq-memprage_")


args               = parser.parse_args()
GROUP              = args.group
subject_id         = args.subject_id
session            = args.session
t1pattern          = args.t1_pattern
overwrite_flag     = bool(args.overwrite)
# Set arguments
mridata            = MRIData(subject_id, session,GROUP,t1pattern)
BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
os.makedirs(ANTS_TRANFORM_PATH,exist_ok=True)
METABOLITE_REF     = args.ref_met
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(args.nthreads)
ftools             = FileTools()

##########################################################



debug.separator()
debug.title(f"Processing {subject_id}-{session}")
############ Denoise PCr+Cr  ##################
__nifti        = mridata.data["mrsi"][METABOLITE_REF]["origfilt"]["nifti"]
if __nifti == 0:
    for metabolite in METABOLITE_LIST:
        debug.info("Denoise Orig",metabolite)
        # metabolite_key = metabolite.replace("+","")
        __nifti_mask   = mridata.data["mrsi"]["mask"]["orig"]["nifti"]
        __nifti        = mridata.data["mrsi"][metabolite]["orig"]["nifti"]
        header_mrsi    = __nifti.header
        brain_mask     = __nifti_mask.get_fdata().squeeze().astype(bool)
        # Filter        
        image_filt_nifti = bhfilt.proc(__nifti,brain_mask)
        mridata.data["mrsi"][metabolite]["origfilt"]["nifti"] = image_filt_nifti
        mrsi_filt_refout_path = mridata.data["mrsi"][metabolite]["orig"]["path"].replace("orig","origfilt")
        ftools.save_nii_file(image_filt_nifti, header_mrsi  ,f"{mrsi_filt_refout_path}")
        mridata.data["mrsi"][metabolite]["origfilt"]["path"] = f"{mrsi_filt_refout_path}"
    debug.success("Done")

############ MRSIto T1w Registration ##################  
transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","mrsi")
transform_prefix          = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w"

transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","mrsi")
transform_dir_prefix_path = join(transform_dir_path,f"{transform_prefix}")
warpfilename              = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w.syn.nii.gz"
if not exists(join(transform_dir_path,warpfilename)) or overwrite_flag:
    debug.warning(f"{METABOLITE_REF} to T1w Registration not found or not up to date")
    syn_tx,_          = reg.register(fixed_input  = mridata.data["t1w"]["brain"]["orig"]["path"],
                                    moving_input  = mridata.data["mrsi"][METABOLITE_REF]["origfilt"]["path"],
                                    fixed_mask    = None, 
                                    moving_mask   = None,
                                    transform     = "sr",
                                    verbose       = 0)
    # Save Transform
    regtools.save_all_transforms(syn_tx,transform_dir_prefix_path)
else:
    debug.success("Already registered")
    #################################
debug.title("DONE")
debug.separator()











    






