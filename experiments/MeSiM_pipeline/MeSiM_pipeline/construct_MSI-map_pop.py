import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split, exists
import os , math, csv
from tools.filetools import FileTools
import nibabel as nib
import pandas as pd
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.nettools import NetTools
from connectomics.parcellate import Parcellate

from tools.mridata import MRIData


import argparse, json


proj_algo = "pca_tsne"

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
    parser.add_argument('--parc', type=str, default="LFMIHIFIS", choices=['LFMIHIFIS','LFIIIIFIS', 'LFMIHIFIF'], 
                        help='Chimera parcellation scheme, choice must be one of: LFMIHIFIS [default], LFMIHIFIF')
    parser.add_argument('--scale',type=int,default=3,help="Cortical parcellation scale (default: 3)")
    parser.add_argument('--group', type=str,default="Mindfulness-Project") 
    parser.add_argument('--proj_comp',type=int, default=1, help='Dim Reduction component (default:1)')
    parser.add_argument('--npert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--dimalg', type=str,default="pca_tsne",choices=['pca_tsne', 'umap'],help='Manifold discovery algorithm (default: pca-tsne)') 
    parser.add_argument('--perplexity', type=float, default=30, help='Perplexity value for t-SNE (default: 30)')
    parser.add_argument('--preproc', type=str, default="filtbiharmonic",help="Preprocessing of orig MRSI files (default: filtbiharmonic)")
    parser.add_argument('--diag', type=str, default="group",
                    help="Only inlcude controls, patients or all ('group'[default])")
    parser.add_argument('--alpha' , type=float, default=0.65 ,
                            help="MRSI-to-T1 parcel coverage above which enough MRSI singal has been recorded (default: 0.65)")
    parser.add_argument('--msiscale' , type=float, default=255.0 ,
                            help="MSI scaling factor, negative values will invert the scales (default: 255)")

    args        = parser.parse_args()
    parc_scheme = args.parc 
    group       = args.group
    dimalg      = args.dimalg
    npert       = args.npert
    perplexity  = args.perplexity
    scale       = args.scale
    proj_comp   = args.proj_comp
    mrsi_cov    = args.alpha
    msiscale    = args.msiscale
    diag        = args.diag
    preproc_string = args.preproc.replace("filt","")

    # OUTDIR PATH
    MSIDIRPATH  = join(dutils.BIDSDATAPATH,group,"derivatives","group","msi","mrsi")     
    os.makedirs(MSIDIRPATH,exist_ok=True)

    


    resultssubdir = join(dutils.BIDSDATAPATH,group,"derivatives","group","connectivity","mrsi")
   
    conFileName    = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_string}_desc-{diag}_connectivity_mrsi.npz"

    con_data       = np.load(join(resultssubdir,conFileName))
    MeSiM_pop_avg  = con_data["MeSiM_pop_avg"]
    parcel_labels  = con_data["parcel_labels_group"]
    parcel_names   = con_data["parcel_names_group"]
    ###############################################################################
    parcellation_data = nib.load(join(dutils.DEVANALYSEPATH,"data","atlas",
                                      f"chimera-{parc_scheme}-{scale}",
                                      f"chimera-{parc_scheme}-{scale}.nii.gz"))
    parcellation_data_np = parcellation_data.get_fdata().astype(int)
    header_mni           = parcellation_data.header

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
        _MeSiM_pop_avg          = np.delete(MeSiM_pop_avg,ignore_rows,axis=0)
        MeSiM_pop_avg           = np.delete(_MeSiM_pop_avg,ignore_rows,axis=1)
        # parcel_concentrations5D = np.delete(parcel_concentrations5D, ignore_rows, axis=1)
        # parcel_concentrations4D = parcel_concentrations5D.mean(axis=0)
    else:
        debug.info("No low coverage parcels detected")


    features_1D          = nettools.dimreduce_matrix(MeSiM_pop_avg,
                                                     method       = dimalg,
                                                     output_dim   = proj_comp,
                                                     perplexity   = perplexity,
                                                     scale_factor = msiscale)

    projected_data_3D = nettools.project_to_3dspace(features_1D,
                                                parcellation_data_np,
                                                parcel_labels)
    
    if dimalg=="pca_tsne":
        dimalg_str     = f"{dimalg}{perplexity}"
    if dimalg=="umap":
        dimalg_str     = f"{dimalg}"
    msiFileName    = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_string}_{dimalg_str}_desc-{diag}_metabsim3D_mrsi.nii.gz"
    outpath        = join(MSIDIRPATH,msiFileName)
    ftools.save_nii_file(projected_data_3D,outpath=outpath,header=header_mni)
    # per parcel MSI
    msiFileName    = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_string}_{dimalg_str}_desc-{diag}_metabsim1D_mrsi.npz"
    np.savez(join(MSIDIRPATH,msiFileName),
                msi_list        = features_1D,
                parcel_labels   = parcel_labels,
                parcel_names    = parcel_names)
    debug.success("MSI map saved to",outpath)




if __name__ == "__main__":
    main()
