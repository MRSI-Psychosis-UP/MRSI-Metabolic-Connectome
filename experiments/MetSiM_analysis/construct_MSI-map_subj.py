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
from connectomics.parcellate import Parcellate
from connectomics.nettools import NetTools
from tools.mridata import MRIData
import argparse, json



dutils    = DataUtils()
nba       = NetBasedAnalysis()
reg       = Registration()
debug     = Debug()
nettools  = NetTools()
ftools    = FileTools()
parc      = Parcellate()

def main():
# Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--parc', type=str, default="LFMIHIFIS", choices=['LFIIIIFIS','LFMIHIFIS', 'LFMIHIFIF','LFMIHISIFF','cubic'], 
                        help='Chimera parcellation scheme, choice must be one of: LFMIHIFIS [default], LFMIHIFIF')
    parser.add_argument('--scale',type=int,default=3,help="Cortical parcellation scale (default: 3)")
    parser.add_argument('--group', type=str,default="Mindfulness-Project") 
    parser.add_argument('--proj_comp',type=int, default=1, help='Dim Reduction component (default:1)')
    parser.add_argument('--npert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--subject_id', type=str, help='subject id', default="S002")
    parser.add_argument('--session', type=str, help='recording session', default="V3")
    parser.add_argument('--perplexity', type=float, default=30, help='Perplexity value for t-SNE (default: 30)')
    parser.add_argument('--preproc', type=str, default="filtbiharmonic_pvcorr_GM",help="Preprocessing of orig MRSI files (default: filtbiharmonic_pvcorr_GM)")
    parser.add_argument('--alpha' , type=float, default=0.65 ,
                                help="MRSI-to-T1 parcel coverage above which enough MRSI singal has been recorded (default: 0.65)")


    args        = parser.parse_args()
    parc_scheme = args.parc 
    group       = args.group
    subject_id  = args.subject_id
    session     = args.session
    npert       = args.npert
    perplexity  = args.perplexity
    scale       = args.scale
    proj_comp   = args.proj_comp
    preproc_string = args.preproc
    mrsi_cov    = args.alpha

    # OUTDIR PATH
    MSIDIRPATH  = join(dutils.BIDSDATAPATH,group,"derivatives","msi",
                    f"sub-{subject_id}",f"ses-{session}")
    debug.info("MSIDIRPATH",MSIDIRPATH)


    os.makedirs(MSIDIRPATH,exist_ok=True)
    mridata             = MRIData(subject_id,session,group=group)
    # Parcellation in MNI space
    metadata             = mridata.extract_metadata(t1mask_path)
    parcellation_data,_  = mridata.get_parcel("mni",parc_scheme,scale,acq=metadata["acq"],run=metadata["run"],grow=growmm)
    parcellation_data_np = parcellation_data.get_fdata().astype(int)
    mni_template         = parcellation_data.header


    connectivity_path = mridata.get_connectivity_path("mrsi",parc_scheme,scale,npert,
                                                      filtoption=preproc_string.replace("filt",""))
    condata           = np.load(connectivity_path)
    MeSiM_subj        = condata["simmatrix_sp"]
    parcel_labels     = condata["labels_indices"]
    simmatrix_ids_to_delete = condata["simmatrix_ids_to_delete"]
    parcel_names      = condata["labels"]
    # Clean matrix:
    # Delete specified rows & columns
    array_after_row_deletion = np.delete(MeSiM_subj, simmatrix_ids_to_delete, axis=0)
    MeSiM_subj       = np.delete(array_after_row_deletion, simmatrix_ids_to_delete, axis=1)
    parcel_labels    = np.delete(parcel_labels,simmatrix_ids_to_delete)
    parcel_names     = np.delete(parcel_names,simmatrix_ids_to_delete)
    ################ Ignore Parcels defined by low MRSI coveraged <-> QMASK ####################
    qmask_dir      = join(dutils.BIDSDATAPATH,group,"derivatives","group","qmask")
    qmask_path     = join(qmask_dir, f"{group}_space-mni_met-CrPCr_desc-qmask_mrsi.nii.gz")
    qmask_pop      = nib.load(qmask_path)
    n_voxel_counts_dict = parc.count_voxels_inside_parcel(qmask_pop.get_fdata(), 
                                                          parcellation_data_np, 
                                                          parcel_labels)
    ignore_parcel_idx = [index for index in n_voxel_counts_dict if n_voxel_counts_dict[index] < mrsi_cov]
    ignore_rows = [np.where(parcel_labels == parcel_idx)[0][0] for parcel_idx in ignore_parcel_idx if len(np.where(parcel_labels == parcel_idx)[0]) != 0]
    ignore_rows = np.sort(np.array(ignore_rows)) 
    for i in ignore_rows:
        debug.info(f"{i} Low MRSI coverage",round(n_voxel_counts_dict[parcel_labels[i]],2),
                   "detected for parcel",parcel_labels[i],parcel_names[i])
    # Delete nodes in MeSiM
    if len(ignore_rows)!=0:
        parcel_labels           = np.delete(parcel_labels,ignore_rows)
        parcel_names            = np.delete(parcel_names,ignore_rows)
        MeSiM_subj              = np.delete(MeSiM_subj,ignore_rows,axis=0)
        MeSiM_subj              = np.delete(MeSiM_subj,ignore_rows,axis=1)
    else:
        debug.info("No low coverage parcels detected")

    # Dim-Reduction and project to MNI
    features_1D  = nettools.dimreduce_matrix(MeSiM_subj,method='pca_tsne',output_dim=proj_comp,perplexity=perplexity,
                                             scale_factor=255.0)

    projected_data_3D = nettools.project_to_3dspace(features_1D,
                                                parcellation_data_np.astype(int),
                                                parcel_labels)
    
    filename  = split(connectivity_path)[1]
    filename  = filename.replace("desc-connectivity_mrsi.npz","desc-3Dmetabsim_mrsi.nii.gz")
    outpath   = join(MSIDIRPATH,filename)
    ftools.save_nii_file(projected_data_3D,header=mni_template,outpath=outpath)
    # per parcel MSI
    np.savez(outpath.replace("3Dmetabsim_mrsi.nii.gz","metabsim_mrsi.npz"),
                msi_list = features_1D,
                labels   = parcel_labels)
    debug.success("MSI map saved to",outpath)




if __name__ == "__main__":
    main()
