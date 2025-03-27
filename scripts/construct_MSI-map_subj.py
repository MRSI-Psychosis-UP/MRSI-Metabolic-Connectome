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
    parser.add_argument('--atlas', type=str, default="LFMIHIFIF-3", choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
                                                                        'LFMIIIFIF-2', 'LFMIIIFIF-3', 'LFMIIIFIF-4'], 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4)')
    parser.add_argument('--group', type=str, default='Dummy-Project', help='Group name (default: "Dummy-Project")')
    parser.add_argument('--proj_comp',type=int, default=1, help='Dim Reduction component (default:1)')
    parser.add_argument('--n_pert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--subject_id',type=str,default="S001",help="Subject ID [sub-??")
    parser.add_argument('--session',type=str,default="V1",help="Session [ses-??")
    parser.add_argument('--perplexity', type=float, default=30, help='Perplexity value for t-SNE (default: 30)')



    args        = parser.parse_args()
    parc_scheme = args.atlas 
    GROUP       = args.group
    subject_id  = args.subject_id
    session     = args.session
    npert       = args.n_pert
    perplexity  = args.perplexity


    proj_comp   = args.proj_comp
    MSIDIRPATH  = join(dutils.BIDSDATAPATH,GROUP,"derivatives","msi",
                    f"sub-{subject_id}",f"ses-{session}","mrsi")
    CONDIRPATH  = join(dutils.BIDSDATAPATH,GROUP,"derivatives","connectomes",
                    f"sub-{subject_id}",f"ses-{session}","mrsi")

    os.makedirs(MSIDIRPATH,exist_ok=True)

    # Parcellation in MNI space
    parcellation_data    = nib.load(join(dutils.DEVDATAPATH,"atlas",f"chimera-{parc_scheme}",f"chimera-{parc_scheme}.nii.gz"))
    parcellation_data_np = parcellation_data.get_fdata().astype(int)
    mni_template         = parcellation_data.header


    filename       = f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_atlas-{parc_scheme}_npert_{npert}_connectivity.npz"
    condata        = np.load(join(CONDIRPATH,filename))
    inputData      = condata["simmatrix_sp"]
    labels_indices = condata["labels_indices"]

    # Dim-Reduction and project to MNI

    features_1D  = nettools.dimreduce_matrix(inputData,method='pca_tsne',output_dim=proj_comp,perplexity=perplexity,
                                             scale_factor=255.0)

    projected_data_3D = nettools.project_to_3dspace(features_1D,
                                                parcellation_data_np.astype(int),
                                                labels_indices)
    
    filename  = f"sub-{subject_id}_ses-{session}_atlas-{parc_scheme}_npert_{npert}_desc-msi3D_mrsi.nii.gz"
    outpath  = join(MSIDIRPATH,filename)
    ftools.save_nii_file(projected_data_3D,mni_template,outpath)
    np.savez(outpath.replace("msi3D_mrsi.nii.gz","msi-parcel_mrsi.npz"),
                msi_list = features_1D,
                labels   = labels_indices)
    debug.success("MSI map saved to",outpath)




if __name__ == "__main__":
    main()
