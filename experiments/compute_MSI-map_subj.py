import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split, exists
import os , math, csv
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from bids.mridata import MRIData
import nibabel as nib
from nilearn import datasets
from connectomics.netcluster import NetCluster
from connectomics.parcellate import Parcellate
from rich.progress import track
import copy, sys
import cupy as cp
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from connectomics.simmilarity import Simmilarity
from connectomics.nettools import NetTools


import argparse, json


proj_algo = "pca_tsne"
ALPHA     = 0.05

dutils    = DataUtils()
nba       = NetBasedAnalysis()
simm      = Simmilarity()
reg       = Registration()
debug     = Debug()
netclust  = NetCluster()
parc      = Parcellate()
simmplt   = SimMatrixPlot() 
nettools  = NetTools()

def main():
# Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--atlas', type=str, default="LFMIHIFIF-3", choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
                                                                        'LFMIIIFIF-2', 'LFMIIIFIF-3', 'LFMIIIFIF-4'], 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4)')
    parser.add_argument('--group', type=str, default='Dummy-Project', help='Group name (default: "Dummy-Project")')
    parser.add_argument('--proj_comp',type=int, default=1, help='Dim Reduction component (default:1)')
    parser.add_argument('--n_pert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--subject_id',type=str,default="S001",help="Subject ID [sub-??")
    parser.add_argument('--session',type=str,default="V1",help="Session [ses-??")



    args        = parser.parse_args()
    parc_scheme = args.atlas 
    GROUP       = args.group
    subject_id  = args.subject_id
    session     = args.session
    npert       = args.n_pert

    ftools      = FileTools(GROUP)
    PROJ_COMP   = args.proj_comp
    MSIDIRPATH  = join(dutils.BIDSDATAPATH,GROUP,"derivatives","msi",
                    f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    CONDIRPATH  = join(dutils.BIDSDATAPATH,GROUP,"derivatives","connectomes",
                    f"sub-{subject_id}",f"ses-{session}","spectroscopy")

    os.makedirs(MSIDIRPATH,exist_ok=True)

    # Parcellation in MNI space
    parcellation_data = nib.load(join(dutils.DEVDATAPATH,"atlas",f"chimera-{parc_scheme}",f"chimera-{parc_scheme}.nii.gz"))
    parcellation_data_np = parcellation_data.get_fdata().astype(int)
    mni_template         = parcellation_data.header


    filename  = f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_atlas-{parc_scheme}_npert_{npert}_connectivity.npz"
    condata   = np.load(join(CONDIRPATH,filename))
    inputData = condata["simmatrix_sp"]
    labels    = condata["labels"]
    labels_indices          = condata["labels_indices"]
    parcel_labels_ignore    = condata["parcel_labels_ignore"]
    simmatrix_ids_to_delete = condata["simmatrix_ids_to_delete"]


    features_1D = nettools.project_matrix(inputData,proj_algo,output_dim=PROJ_COMP)
    projected_data_3D = simm.nodal_strength_map(features_1D,
                                                parcellation_data_np.astype(int),
                                                labels_indices)
    filename  = f"sub-{subject_id}_ses-{session}_atlas-{parc_scheme}_npert_{npert}_desc-msi3D.nii.gz"
    outpath  = join(MSIDIRPATH,filename)
    ftools.save_nii_file(projected_data_3D,mni_template,outpath)
    np.savez(outpath.replace("msi3D.nii.gz","msi_parcel.npz"),
                msi_list = features_1D,
                labels   = labels_indices)




if __name__ == "__main__":
    main()
