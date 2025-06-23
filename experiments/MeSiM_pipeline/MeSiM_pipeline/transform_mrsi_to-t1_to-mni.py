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
import nibabel as nib
from rich.progress import Progress, TaskID
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed


dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
ftools   = FileTools()


def transform_worker(fixed_image, moving_image, transform_list, outpath):
    try:
        out_nifti = reg.transform(fixed_image, moving_image, transform_list).to_nibabel()
        os.makedirs(split(outpath)[0], exist_ok=True)
        ftools.save_nii_file(out_nifti, outpath=outpath)
        return outpath  # success marker
    except Exception as e:
        return {"error": str(e), "outpath": outpath}

def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    # Parse arguments
    parser.add_argument('--group', type=str,default="Mindfulness-Project") 
    parser.add_argument('--subject_id',type=str,default="S002",help="Subject ID [sub-??")
    parser.add_argument('--session',type=str,default="V3",help="Session [ses-??")
    parser.add_argument('--nthreads',type=int,default=4,help="Number of CPU threads [default=4]")
    parser.add_argument('--filtoption',type=str,default="filtbiharmonic",help="MRSI filter option  [default=filtbihamonic]")
    parser.add_argument('--overwrite' , type=int, default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--t1'        , type=str, default=None,help="Anatomical T1w file path")
    parser.add_argument('--b0'        , type=float, default = 3,choices=[3,7],help="MRI B0 field strength in Tesla [default=3]")



    args               = parser.parse_args()
    GROUP              = args.group
    filtoption         = args.filtoption
    overwrite          = args.overwrite
    t1_path_arg        = args.t1
    B0_strength        = args.b0
    nthreads           = args.nthreads

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
    subject_id, session = args.subject_id, args.session
    recording_id = f"sub-{subject_id}_ses-{session}"

    BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,GROUP)
    ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
    METABOLITE_LIST = ["CrPCr", "GluGln", "GPCPCh", "NAANAAG", "Ins"]

    if B0_strength == 3:
        METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]
    elif B0_strength == 7:
        METABOLITE_LIST = [
            "NAA", "NAAG", "Ins", "GPCPCh", "Glu", "Gln", "CrPCr", "GABA", "GSH"]


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

    debug.title(f"Transforming {recording_id}")

    #################### MRSI-orig --> MRSI-anat ####################
    debug.info("MRSI-orig --> MRSI-anat")
    transform_list   = mridata.get_transform("forward", "mrsi")
    with Progress() as progress:
        task = progress.add_task("Transforming...", total=len(SIGNAL_LIST))
        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            futures = []
            for component in SIGNAL_LIST:
                met, desc, option = component
                try:
                    
                    mrsi_img_orig_nifti = mridata.get_mri_nifti(
                        modality="mrsi", space="orig", desc=desc, met=met, option=option
                    )
                    mrsi_img_anat_path = mridata.get_mri_filepath(
                        modality="mrsi", space="anat", desc=desc, met=met, option=option
                    )
                    futures.append(executor.submit(
                        transform_worker,
                        t1_path,
                        mrsi_img_orig_nifti,
                        transform_list,
                        mrsi_img_anat_path
                    ))
                except Exception as e:
                    debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                    progress.advance(task)

            # As each job completes, update the progress bar
            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, dict) and "error" in result:
                    debug.error(f"Transform failed: {result['outpath']}", result["error"])
                progress.update(task, description=f"Collecting results")
                progress.advance(task)


    #################### MRSI-anat --> MRSI-MNI ####################
    debug.info("MRSI-anat --> MRSI-MNI")
    t1_resolution    = np.array(nib.load(t1_path).header.get_zooms()[:3]).mean()
    mni_ref          = datasets.load_mni152_template(t1_resolution)
    transform_list   = mridata.get_transform("forward", "anat")
    with Progress() as progress:
        task = progress.add_task("Transforming...", total=len(SIGNAL_LIST))
        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            futures = []
            for component in SIGNAL_LIST:
                met, desc, option = component
                try:
                    mrsi_anat_nifti = mridata.get_mri_nifti(
                        modality="mrsi", space="anat", desc=desc, met=met, option=option
                    )
                    mrsi_img_mni_path = mridata.get_mri_filepath(
                        modality="mrsi", space="mni", desc=desc, met=met, option=option
                    )
                    futures.append(executor.submit(
                        transform_worker,
                        mni_ref,
                        mrsi_anat_nifti,
                        transform_list,
                        mrsi_img_mni_path
                    ))
                except Exception as e:
                    debug.error(f"Error preparing task: {recording_id} - {met, desc, option} Exception", e)
                    progress.advance(task)

            # As each job completes, update the progress bar
            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, dict) and "error" in result:
                    debug.error(f"Transform failed: {result['outpath']}", result["error"])
                progress.update(task, description=f"Collecting results")
                progress.advance(task)

    debug.success(f"DONE {recording_id}")
    debug.separator()
    return

if __name__=="__main__":
    main()














    






