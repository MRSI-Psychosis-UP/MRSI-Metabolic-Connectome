import numpy as np
from mrsitoolbox.tools.datautils import DataUtils
from mrsitoolbox.tools.debug import Debug
from os.path import join, split, exists
from mrsitoolbox.tools.filetools import FileTools
import nibabel as nib
from mrsitoolbox.connectomics.network import NetBasedAnalysis
from mrsitoolbox.connectomics.nettools import NetTools
import argparse



dutils    = DataUtils()
nba       = NetBasedAnalysis()
debug     = Debug()
nettools  = NetTools()
 

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--atlas', type=str, default='LFMIHIFIF-3', choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
                                                                     'LFMIIIFIF-2', 'LFMIIIFIF-3', 'LFMIIIFIF-4', 
                                                                     'geometric_cubeK18mm','geometric_cubeK23mm',
                                                                     'aal', 'destrieux','mist-197','schaefer-200'], 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4, geometric, aal, destrieux)')
    parser.add_argument('--group', type=str, default='Geneva-Study', help='Group name (default: "Geneva-Study")')
    parser.add_argument('--perplexity', type=float, default=50, help='Perplexity value for t-SNE (default: 50)')

    

    args        = parser.parse_args()
    parc_scheme = args.atlas 
    group       = args.group
    perplexity  = args.perplexity
    ftools      = FileTools()

    ############## GET MNI Parcellation ###############
    parcellation_path    = join(dutils.DEVDATAPATH,"atlas",f"chimera-{parc_scheme}",f"chimera-{parc_scheme}.nii.gz")
    parcellation_data_np = nib.load(parcellation_path).get_fdata()
    header_mni           = nib.load(parcellation_path).header

    # Load MeSiM
    mesim_path    = join(dutils.BIDSDATAPATH,group,"derivatives","group","MeSiM",
                         f"metabolic_similarity_matrix_atlas-{parc_scheme}_mrsi.npz")
    data          = np.load(mesim_path)
    label_indices = data["label_indices"]
    mesim_group   = data["features_ND_group"]

    
    # Dim-Reduction and project to MNI
    features_1D_group  = nettools.dimreduce_matrix(mesim_group,method='pca_tsne',output_dim=1,perplexity=perplexity)

    projected_data_3D = nettools.project_to_3dspace(features_1D_group,
                                            parcellation_data_np,
                                            label_indices)
    msi_path   = join(dutils.BIDSDATAPATH,f"{group}","derivatives","group","metabolic-similarity-map",
                        f"metabolic-similarity-map_space-mni_atlas-{parc_scheme}_proj-pca-tsne_mrsi.nii.gz")
    ftools.save_nii_file(projected_data_3D,header_mni,msi_path)
    debug.info("MSI saved to",msi_path)


if __name__=="__main__":
    main()