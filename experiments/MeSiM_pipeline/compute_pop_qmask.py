import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import copy
from rich.progress import Progress
import pandas as pd
from tools.datautils import DataUtils
from tools.datautils import DataUtils
from graphplot.slices import PlotSlices
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
import nibabel as nib
import argparse
from os.path import join, split
from nilearn import datasets
from tools.mridata import MRIData
import json

dutils   = DataUtils()
ftools   = FileTools()
debug    = Debug()


def main():
    ###############################################################################
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    # Parse arguments
    parser.add_argument('--group', type=str,default="Mindfulness-Project") 
    parser.add_argument('--overwrite' , type=int, default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--participants', type=str, default=None,
                        help="Path to TSV file containing list of participant IDs and sessions to include. If not specified, process all.")
    parser.add_argument('--snr' , type=float, default=4 ,help="SNR threshold above which an MRSI signal is deemed signficant (default: 4)")
    parser.add_argument('--crlb' , type=float, default=20 ,help="CRLB threshold below which an MRSI signal is deemed signficant (default: 4)")
    parser.add_argument('--fwhm' , type=float, default=0.1 ,help="FWHM threshold below which an MRSI signal is deemed signficant (default: 4)")
    parser.add_argument('--alpha' , type=float, default=0.68 ,
                        help="Proportion of significant voxels among n participants above which an MRSI voxel is deemed signifcant at a group level (default: 0.68)")
    parser.add_argument('--b0'        , type=float, default = 3,choices=[3,7],help="MRI B0 field strength in Tesla [default=3]")


    args               = parser.parse_args()
    #
    group              = args.group
    overwrite          = args.overwrite
    participants_file  = args.participants
    B0_strength        = args.b0

    # Quality params
    snr_th      = args.snr
    fwhm_th     = args.fwhm
    crlb_th     = args.crlb
    qmaskpop_th = args.alpha
    #############################


    if participants_file is None:
        participant_session_list = join(dutils.BIDSDATAPATH,group,"participants_allsessions.tsv")
        df                       = pd.read_csv(participant_session_list, sep='\t')
        df = df[df.session_id != "V2BIS"]
    else:
        df = pd.read_csv(participants_file, sep='\t')
        
    subject_id_list = df.participant_id.to_list()
    session_id_list = df.session_id.to_list()

    ###############################

    BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,group)
    OUTDIR             = join(BIDS_ROOT_PATH,"derivatives","group","qmask")

    if B0_strength == 3:
        METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]
    elif B0_strength == 7:
        METABOLITE_LIST = ["NAA", "NAAG", "Ins", "GPCPCh", "Glu", "Gln", "CrPCr", "GABA", "GSH"]



    ############ Create QMASK template ##################
    mridata = MRIData(subject_id_list[0],session_id_list[0],group)
    _shape  = mridata.get_mri_nifti("mrsi","mni","crlb",met=METABOLITE_LIST[0]).shape
    met_qmask = np.zeros((len(METABOLITE_LIST),)+_shape)

    count = 0
    with Progress() as progress:
        task = progress.add_task("Starting...", total=len(subject_id_list))
        
        for subject_id, session in zip(subject_id_list, session_id_list):
            # Update the progress bar description with current subject and session.
            progress.update(task, description=f"Processing Subject: {subject_id}, Session: {session}")
            try:
                mridata = MRIData(subject_id, session, group)
                snr = mridata.get_mri_nifti("mrsi", "mni", "snr").get_fdata()
                fwhm = mridata.get_mri_nifti("mrsi", "mni", "fwhm").get_fdata()
                mask = np.ones(snr.shape)
                mask[snr < snr_th] = 0
                mask[fwhm > fwhm_th] = 0
                for idm, metabolite in enumerate(METABOLITE_LIST):
                    crlb = mridata.get_mri_nifti("mrsi", "mni", "crlb", met=metabolite).get_fdata()
                    mask[crlb > crlb_th] = 0
                    met_qmask[idm] += mask
                count += 1
            except Exception as e:
                debug.warning("No", subject_id, session, "found:", e)
            # Advance the progress bar
            progress.advance(task)

    met_qmask /= count
    debug.success("Averaged ",count,"recordings")
    met_qmask_pop = copy.deepcopy(met_qmask)
    met_qmask_pop[met_qmask<=qmaskpop_th] = 0
    met_qmask_pop[met_qmask>qmaskpop_th]  = 1

    mni_template    = datasets.load_mni152_template()
    os.makedirs(OUTDIR,exist_ok=True)
    for idm, metabolite in enumerate(METABOLITE_LIST):
        outpath = join(OUTDIR, f"{group}_space-mni_met-{metabolite}_desc-qmask_mrsi.nii.gz")
        ftools.save_nii_file(met_qmask_pop[idm],outpath,mni_template.header)
        debug.success("QMASK Saved to",outpath)


            
        #################################
    debug.title("DONE")
    debug.separator()

if __name__=="__main__":
    main()









    






