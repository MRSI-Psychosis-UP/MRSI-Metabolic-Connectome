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
parser.add_argument('--t1'        , type=str, default=None,help="Anatomical T1w file path")


args               = parser.parse_args()
GROUP              = args.group
subject_id         = args.subject_id
session            = args.session
t1_path_arg        = args.t1
overwrite_flag     = bool(args.overwrite)
# Set arguments
mridata            = MRIData(subject_id, session,GROUP)
BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
os.makedirs(ANTS_TRANFORM_PATH,exist_ok=True)
METABOLITE_REF     = args.ref_met
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(args.nthreads)
ftools             = FileTools()

##########################################################

# mridata.get_mri_filepath(modality="t1w",space="orig",desc="brain")

debug.separator()
debug.title(f"Processing {subject_id}-{session}")
############ Denoise PCr+Cr  ##################
__path   = mridata.get_mri_filepath(modality="mrsi",space="orig", desc="signal", met=METABOLITE_REF,option="filtbiharmonic")
if not exists(__path) :
    __nifti_mask   = mridata.get_mri_nifti(modality="mrsi",space="orig",desc="brainmask")
    brain_mask     = __nifti_mask.get_fdata().squeeze().astype(bool)
    header_mrsi    = __nifti_mask.header
    for metabolite in METABOLITE_LIST:
        debug.info("Denoise Orig",metabolite)
        image_og_nifti        = mridata.get_mri_nifti(modality="mrsi",space="orig",desc="signal",met=metabolite)       
        # Filter        
        image_filt_nifti      = bhfilt.proc(image_og_nifti,brain_mask,fwhm=6,percentile=98)
        mrsi_filt_refout_path = mridata.get_mri_filepath(modality="mrsi",space="orig",desc="signal",
                                                         met=metabolite,option="filtbiharmonic")
        ftools.save_nii_file(image_filt_nifti, header_mrsi  ,f"{mrsi_filt_refout_path}")
    debug.success("Done")

############ MRSIto T1w Registration ##################  
if t1_path_arg:
    t1_path = t1_path_arg
else:
    t1_path = mridata.get_mri_nifti(modality="t1w",space="orig",desc="brain")
# Load transform paths
transform_prefix          = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w"
transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","mrsi")
transform_dir_prefix_path = join(transform_dir_path,f"{transform_prefix}")
warpfilename              = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w.syn.nii.gz"
if not exists(join(transform_dir_path,warpfilename)) or overwrite_flag:
    debug.warning(f"{METABOLITE_REF} to T1w Registration not found or not up to date")
    syn_tx,_          = reg.register(fixed_input   = t1_path,
                                     moving_input  = mridata.get_mri_nifti(modality="mrsi",space="orig",desc="signal",
                                                                          met=METABOLITE_REF, option="filtbiharmonic"), 
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











    






