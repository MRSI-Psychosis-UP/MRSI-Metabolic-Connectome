import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split, exists
import os , math, csv
from tools.filetools import FileTools
import nibabel as nib
import copy, sys
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.nettools import NetTools
from tools.mridata import MRIData


import argparse, json


proj_algo = "pca_tsne"

dutils    = DataUtils()
nba       = NetBasedAnalysis()
reg       = Registration()
debug     = Debug()
nettools  = NetTools()
ftools    = FileTools()

def main():
# Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--parc', type=str, default="LFMIHIFIS", choices=['LFMIHIFIS', 'LFMIHIFIF'], 
                        help='Chimera parcellation scheme, choice must be one of: LFMIHIFIS [default], LFMIHIFIF')
    parser.add_argument('--scale',type=int,default=3,help="Cortical parcellation scale (default: 3)")
    parser.add_argument('--group', type=str,default="Mindfulness-Project") 
    parser.add_argument('--proj_comp',type=int, default=1, help='Dim Reduction component (default:1)')
    parser.add_argument('--npert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--subject_id', type=str, help='subject id', default="S002")
    parser.add_argument('--session', type=str, help='recording session',choices=['V1', 'V2', 'V3','V4','V5'], default="V3")
    parser.add_argument('--perplexity', type=float, default=30, help='Perplexity value for t-SNE (default: 30)')
    parser.add_argument('--preproc', type=str, default="filtbiharmonic",help="Preprocessing of orig MRSI files (default: filtbiharmonic)")



    args        = parser.parse_args()
    parc_scheme = args.parc 
    GROUP       = args.group
    subject_id  = args.subject_id
    session     = args.session
    npert       = args.npert
    perplexity  = args.perplexity
    scale       = args.scale
    proj_comp   = args.proj_comp
    preproc_string = args.preproc

    # OUTDIR PATH
    MSIDIRPATH  = join(dutils.BIDSDATAPATH,GROUP,"derivatives","msi",
                    f"sub-{subject_id}",f"ses-{session}")
    debug.info("MSIDIRPATH",MSIDIRPATH)


    os.makedirs(MSIDIRPATH,exist_ok=True)
    mridata             = MRIData(subject_id,session,group=GROUP)
    # Parcellation in MNI space
    parcellation_data,_  = mridata.get_parcel("mni",parc_scheme,scale)
    parcellation_data_np = parcellation_data.get_fdata().astype(int)
    mni_template         = parcellation_data.header


    connectivity_path = mridata.get_connectivity_path("mrsi",parc_scheme,scale,npert,
                                                      filtoption=preproc_string.replace("filt",""))
    condata           = np.load(connectivity_path)
    inputData         = condata["simmatrix_sp"]
    labels_indices    = condata["labels_indices"]
    simmatrix_ids_to_delete = condata["simmatrix_ids_to_delete"]
    # Clean matrix:
    # Delete specified rows & columns
    array_after_row_deletion = np.delete(inputData, simmatrix_ids_to_delete, axis=0)
    inputData       = np.delete(array_after_row_deletion, simmatrix_ids_to_delete, axis=1)
    labels_indices  = np.delete(labels_indices,simmatrix_ids_to_delete)

    # Dim-Reduction and project to MNI

    features_1D  = nettools.dimreduce_matrix(inputData,method='pca_tsne',output_dim=proj_comp,perplexity=perplexity,
                                             scale_factor=255.0)

    projected_data_3D = nettools.project_to_3dspace(features_1D,
                                                parcellation_data_np.astype(int),
                                                labels_indices)
    
    filename  = split(connectivity_path)[1]
    filename  = filename.replace("desc-connectivity_mrsi.npz","desc-3Dmetabsim_mrsi.nii.gz")
    outpath   = join(MSIDIRPATH,filename)
    ftools.save_nii_file(projected_data_3D,header=mni_template,outpath=outpath)
    # per parcel MSI
    np.savez(outpath.replace("3Dmetabsim_mrsi.nii.gz","metabsim_mrsi.npz"),
                msi_list = features_1D,
                labels   = labels_indices)
    debug.success("MSI map saved to",outpath)




if __name__ == "__main__":
    main()
