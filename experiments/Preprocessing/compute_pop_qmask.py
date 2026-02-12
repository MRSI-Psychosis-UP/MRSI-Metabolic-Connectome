import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import copy
from rich.progress import Progress
import pandas as pd
from mrsitoolbox.tools.datautils import DataUtils
from mrsitoolbox.tools.datautils import DataUtils
from os.path import split, join, exists
from mrsitoolbox.tools.filetools import FileTools
from mrsitoolbox.tools.debug import Debug
import nibabel as nib
import argparse
from os.path import join, split
from nilearn import datasets
from mrsitoolbox.tools.mridata import MRIData
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
    parser.add_argument('--fwhm' , type=float, default=0.1 ,help="FWHM threshold below which an MRSI signal is deemed signficant (default: 0.1)")
    parser.add_argument('--alpha' , type=float, default=0.68 ,
                        help="Proportion of significant voxels among n participants above which an MRSI voxel is deemed signifcant at a group level (default: 0.68)")
    parser.add_argument('--b0'        , type=float, default = 3,choices=[3,7],help="MRI B0 field strength in Tesla [default=3]")
    parser.add_argument('--res' , type=str, default="5", help="MRSI spatial isotropic resolution in mm (default 5)")
    parser.add_argument('--select','-s', dest='select', action='append', default=[],
                        help="Filter participants by covariate in the TSV. Can be repeated. Format: COVARNAME,VALUE or COVARNAME,>VALUE (e.g., --select Diag,1 --select Age,>12)")

    args               = parser.parse_args()
    #
    group              = args.group
    overwrite          = args.overwrite
    participants_file  = args.participants
    B0_strength        = args.b0
    res                = args.res
    debug.info("select",args.select)
    # Quality params
    snr_th      = args.snr
    fwhm_th     = args.fwhm
    crlb_th     = args.crlb
    qmaskpop_th = args.alpha
    #############################
    debug.info("snr",snr_th)
    debug.info("fwhm_th",fwhm_th)
    debug.info("crlb_th",crlb_th)

    if participants_file is None:
        participants_file = join(dutils.BIDSDATAPATH,group,"participants_allsessions.tsv")
        df                       = pd.read_csv(participants_file, sep='\t')
        df = df[df.session_id != "V2BIS"]
    else:
        df = pd.read_csv(participants_file, sep='\t')

    covariate_names = df.columns.to_list()
    selection_suffix = ""


    selections = args.select or []
    if selections:
        selection_suffix_parts = []
        mask = pd.Series(True, index=df.index)

        for raw_select in selections:
            try:
                covar_name, condition_raw = [part.strip() for part in raw_select.split(",", 1)]
            except ValueError:
                debug.error(f"Invalid --select format '{raw_select}'. Use COVARNAME,VALUE (e.g., --select Diag,1 or --select Age,>12)")
                sys.exit(1)

            if covar_name not in covariate_names:
                debug.error(f"Covariate '{covar_name}' not found in participants file. Available covariates: {covariate_names}")
                sys.exit(1)

            covar_series = df[covar_name]
            operator = "=="
            value_str = condition_raw
            for candidate_op in [">=", "<=", "!=", ">", "<"]:
                if condition_raw.startswith(candidate_op):
                    operator = candidate_op
                    value_str = condition_raw[len(candidate_op):].strip()
                    break
            else:
                if condition_raw.startswith("=="):
                    operator = "=="
                    value_str = condition_raw[2:].strip()

            if value_str == "":
                debug.error(f"Missing comparison value in selection '{raw_select}'.")
                sys.exit(1)

            needs_numeric = operator in [">", "<", ">=", "<="]
            is_numeric = pd.api.types.is_numeric_dtype(covar_series)

            if needs_numeric:
                try:
                    covar_series_numeric = pd.to_numeric(covar_series)
                except Exception:
                    debug.error(f"Covariate '{covar_name}' must be numeric for comparison with '{operator}'.")
                    sys.exit(1)
                try:
                    covar_value = pd.to_numeric(value_str)
                except ValueError:
                    debug.error(f"Covariate '{covar_name}' expects numeric values; could not parse '{value_str}'.")
                    sys.exit(1)
                compare_series = covar_series_numeric
            else:
                if is_numeric:
                    try:
                        covar_value = pd.to_numeric(value_str)
                    except ValueError:
                        debug.error(f"Covariate '{covar_name}' expects numeric values; could not parse '{value_str}'.")
                        sys.exit(1)
                else:
                    if operator not in ["==", "!="]:
                        debug.error(f"Covariate '{covar_name}' is non-numeric; only == or != comparisons are supported.")
                        sys.exit(1)
                    covar_value = value_str
                compare_series = covar_series

            if operator == "==":
                mask &= compare_series == covar_value
            elif operator == "!=":
                mask &= compare_series != covar_value
            elif operator == ">":
                mask &= compare_series > covar_value
            elif operator == "<":
                mask &= compare_series < covar_value
            elif operator == ">=":
                mask &= compare_series >= covar_value
            elif operator == "<=":
                mask &= compare_series <= covar_value
            else:
                debug.error(f"Unsupported operator '{operator}' in selection '{raw_select}'.")
                sys.exit(1)

            selection_suffix_parts.append(f"{covar_name.replace(' ', '-')}-{operator}{value_str}")
            debug.info(f"Applying participant filter {covar_name} {operator} {covar_value}; {mask.sum()} rows remain")

        df = df[mask]
        if df.empty:
            debug.error(f"No participants matched filters: {selections}")
            sys.exit(1)
        selection_suffix = "_sel-" + "_".join(selection_suffix_parts)

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
    debug.info("res",res)
    mridata = MRIData(subject_id_list[0],session_id_list[0],group) 
    _tmp  = mridata.get_mri_nifti("mrsi","mni","crlb",
                                    met=METABOLITE_LIST[0],
                                    res=res)
    shape = _tmp.shape
    header = _tmp.header
    met_qmask = np.zeros((len(METABOLITE_LIST),)+shape)
    count = 0
    met_qmask_subj = np.zeros((len(METABOLITE_LIST),)+shape+(len(session_id_list),))
    with Progress() as progress:
        task = progress.add_task("Starting...", total=len(subject_id_list))
        for ids, (subject_id, session) in enumerate(zip(subject_id_list, session_id_list)):
            # Update the progress bar description with current subject and session.
            progress.update(task, description=f"Processing Subject: {subject_id}, Session: {session}")
            try:
                mridata = MRIData(subject_id, session, group)
                snr  = mridata.get_mri_nifti("mrsi", "mni", "snr",res=res).get_fdata()
                fwhm = mridata.get_mri_nifti("mrsi", "mni", "fwhm",res=res).get_fdata()
                for idm, metabolite in enumerate(METABOLITE_LIST):
                    crlb = mridata.get_mri_nifti("mrsi", "mni", "crlb", 
                                                    met=metabolite,res=res).get_fdata()

                    qflag = (snr > snr_th) & (fwhm < fwhm_th) & (crlb < crlb_th)
                    mask = qflag.astype(np.float32)
                    met_qmask[idm]      += mask
                    met_qmask_subj[idm,:,:,:,ids] = mask
                count += 1
            except Exception as e:
                debug.warning("No", subject_id, session, "found:", e)
            progress.advance(task)

    met_qmask /= count # average over all the recordings
    debug.success("Averaged ",count,"recordings")
    met_qmask_pop = np.ones_like(met_qmask)
    met_qmask_pop[met_qmask<=qmaskpop_th] = 0
    met_qmask_pop[met_qmask>qmaskpop_th]  = 1

    mni_template    = datasets.load_mni152_template(int(res))
    os.makedirs(OUTDIR,exist_ok=True)
    for idm, metabolite in enumerate(METABOLITE_LIST):
        outpath = join(OUTDIR, f"{group}_space-mni_met-{metabolite}_desc-qmask_mrsi.nii.gz")
        ftools.save_nii_file(met_qmask_pop[idm],outpath,header)
        debug.success("QMASK Saved to",outpath)
        outpath = join(OUTDIR, f"{group}4D_space-mni_met-{metabolite}_desc-qmask_mrsi.nii.gz")
        ftools.save_nii_file(met_qmask_subj[idm],outpath,header)

    
    
    
    for ids, (subject_id, session) in enumerate(zip(subject_id_list, session_id_list)):
        debug.info(subject_id, session)


            
        #################################
    debug.title("DONE")
    debug.separator()

if __name__=="__main__":
    main()









    




