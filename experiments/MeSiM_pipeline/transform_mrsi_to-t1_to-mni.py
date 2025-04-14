import os, sys, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from registration.registration import Registration
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
from tools.mridata import MRIData
from nilearn import datasets
from rich.progress import Progress


dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
mni_ref  = datasets.load_mni152_template()

def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    # Parse arguments
    parser.add_argument('--group', type=str,default="Mindfulness-Project") 
    parser.add_argument('--subject_id',type=str,default="S002",help="Subject ID [sub-??")
    parser.add_argument('--session',type=str,default="V3",help="Session [ses-??")
    parser.add_argument('--nthreads',type=int,default=4,help="Number of CPU threads [default=4]")
    parser.add_argument('--t1_pattern',type=str,default="_run-01_acq-memprage_",help="T1w file pattern e.g _run-01_acq-memprage_")
    parser.add_argument('--filtoption',type=str,default="filtbiharmonic",help="MRSI filter option  [default=filtbihamonic]")
    parser.add_argument('--overwrite' , type=int, default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--t1'        , type=str, default=None,help="Anatomical T1w file path")



    args               = parser.parse_args()
    GROUP              = args.group
    t1pattern          = args.t1_pattern
    filtoption         = args.filtoption
    overwrite          = args.overwrite
    t1_path_arg        = args.t1

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(args.nthreads)
    ftools   = FileTools()
    subject_id, session = args.subject_id, args.session
    recording_id = f"sub-{subject_id}_ses-{session}"

    BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,GROUP)
    ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
    METABOLITE_LIST = ["CrPCr", "GluGln", "GPCPCh", "NAANAAG", "Ins"]

    # Use a list comprehension to generate two entries per metabolite:
    SIGNAL_LIST = [
        [met, "signal", filtoption] if entry == "signal" else [met, "crlb", None]
        for met in METABOLITE_LIST
        for entry in ("signal", "crlb")
    ] + [
        ["water", "signal", None],
        [None, "snr", None],
        [None, "fwhm", None],
        [None, "brainmask", None]
    ]
    debug.title(f"Processing {recording_id}")
    mridata = MRIData(subject_id, session,GROUP)

    # Check if already processed
    if t1_path_arg:
        t1_path = t1_path_arg
    else:
        t1_path = mridata.get_mri_filepath(modality="t1w",space="orig",desc="brain")

    if not exists(t1_path):
        debug.error(f"{t1_path} path does not exist")
        return

    # Check if already processed
    __path =  mridata.get_mri_filepath(modality="mrsi",space="mni",desc="signal",met=METABOLITE_LIST[0],option=filtoption)
    if exists(__path) and not overwrite:
        debug.info("Already transformed: SKIP")
        return

    with Progress() as progress:
        task = progress.add_task("Transforming...", total=len(SIGNAL_LIST))
        for idm, component in enumerate(SIGNAL_LIST):
            met, desc, option = component
            # Update progress bar description with current component details.
            
            try:
                progress.update(task, description=f"Transforming: {met}, {desc}, {option} to T1W")
                # Get the original MRSI image in NIfTI format.
                mrsi_img_orig_nifti = mridata.get_mri_nifti(modality="mrsi", space="orig", desc=desc, met=met, option=option)
                # Transform the MRSI image into T1w space.
                transform_list   = mridata.get_transform("forward", "mrsi")
                mrsi_anat_nifti  = reg.transform(t1_path, mrsi_img_orig_nifti, transform_list).to_nibabel()
                # Save MRSI-T1w image.
                mrsi_img_anat_path = mridata.get_mri_filepath(modality="mrsi", space="t1w", desc=desc, met=met, option=option)
                os.makedirs(split(mrsi_img_anat_path)[0], exist_ok=True)
                ftools.save_nii_file(mrsi_anat_nifti, outpath=mrsi_img_anat_path)

                # Transform the MRSI-T1w image into MNI space.
                progress.update(task, description=f"Transforming: {met}, {desc}, {option} to MNI")
                transform_list   = mridata.get_transform("forward", "anat")
                mrsi_mni_nifti   = reg.transform(mni_ref, mrsi_anat_nifti, transform_list).to_nibabel()
                # Save MRSI-MNI image.
                mrsi_img_mni_path = mridata.get_mri_filepath(modality="mrsi", space="mni", desc=desc, met=met, option=option)
                os.makedirs(split(mrsi_img_mni_path)[0], exist_ok=True)
                ftools.save_nii_file(mrsi_mni_nifti, outpath=mrsi_img_mni_path)
            except Exception as e:
                debug.error(f"{recording_id} - {met, desc, option} Exception", e)
            progress.advance(task)

    debug.success(f"DONE {recording_id}")
    debug.separator()
    return


if __name__=="__main__":
    main()














    






