import os, sys, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from registration.registration import Registration
from registration.tools import RegTools
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug

from os.path import join, split
from tools.mridata import MRIData
import json


dutils   = DataUtils()
debug    = Debug()
regtools = RegTools()
reg      = Registration()


parser = argparse.ArgumentParser(description="Process some input parameters.")
# Parse arguments
parser.add_argument('--group', type=str,default="Dummy-Project") 
parser.add_argument('--subject_id',type=str,default="S001",help="Subject ID [sub-??")
parser.add_argument('--session',type=str,default="V1",help="Session [ses-??")
parser.add_argument('--nthreads',type=int,default=4,help="Number of CPU threads [default=4]")
parser.add_argument('--t1_pattern',type=str,default="_run-01_acq-memprage_",help="T1w file pattern e.g _run-01_acq-memprage_")

args               = parser.parse_args()
GROUP              = args.group
t1pattern          = args.t1_pattern


os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(args.nthreads)
ftools   = FileTools()
subject_id, session = args.subject_id, args.session
recording_id = f"sub-{subject_id}_ses-{session}"

BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins",
                      "CrPCr-crlb","GluGln-crlb","GPCPCh-crlb","NAANAAG-crlb","Ins-crlb",
                      "snr","fwhm","mask"]



debug.title(f"Processing {recording_id}")
mridata = MRIData(subject_id, session,GROUP,t1pattern)
outpath = mridata.get_path("spectroscopy", "mask", "mni")

mridata.load_t1w(t1pattern)
for idm, metabolite in enumerate(METABOLITE_LIST):
    # Transform to anat space
    outpath = mridata.get_path("spectroscopy", metabolite, "t1w")
    mrsi_t1wspace_nifti = mridata.get_mrsi_volume(metabolite, "t1w")
    if outpath is not None and mrsi_t1wspace_nifti is not None:
        ftools.save_nii_file(mrsi_t1wspace_nifti.get_fdata(), mrsi_t1wspace_nifti.header, f"{outpath}")
    else:
        debug.error(recording_id, "outpath does not exist", outpath)

    # Transform to mni space
    outpath = mridata.get_path("spectroscopy", metabolite, "mni")
    mrsi_mni_nifti = mridata.get_mrsi_volume(metabolite, "mni")
    if outpath is not None and mrsi_mni_nifti is not None:
        ftools.save_nii_file(mrsi_mni_nifti.get_fdata(), mrsi_mni_nifti.header, f"{outpath}")
    else:
        debug.error(recording_id, "outpath does not exist", outpath)

debug.success(f"DONE {recording_id}")
debug.separator()








        
    #################################












    






