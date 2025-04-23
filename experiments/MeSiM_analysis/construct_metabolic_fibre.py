import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, image, datasets
from scipy import interpolate
import copy, sys
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from graphplot.colorbar import ColorBar
import pandas as pd
import umap
from connectomics import umap_con

import argparse
colorbar    = ColorBar()

dutils    = DataUtils()       
debug     = Debug()
FONTSIZE         = 16
confInterval                      = 0.95


parser = argparse.ArgumentParser(description="Process some input parameters.")
parser.add_argument('--group', type=str,default="Mindfulness-Project") 
parser.add_argument('--parc', type=str, default="LFMIHIFIS",
                    help='Chimera parcellation scheme, valid choices: LFMIHIFIS, LFMIHIFIF. Default: LFMIHIFIS')
parser.add_argument('--scale', type=int, default=3,
                    help="Cortical parcellation scale (default: 3)")
parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1],
                    help="Overwrite existing parcellation (default: 0)")
parser.add_argument('--npert', type=int, default=50,
                    help='Number of perturbations as comma-separated integers (default: 50)')
parser.add_argument('--perplexity', type=float, default=30, help='Perplexity value for t-SNE (default: 30)')
parser.add_argument('--dimalg', type=str,default="pca_tsne",choices=['pca_tsne', 'umap'],help='Manifold discovery algorithm (default: pca-tsne)') 
parser.add_argument('--diag', type=str, default="group",
                help="Only inlcude controls, patients or all ('group'[default])")
parser.add_argument('--preproc', type=str, default="filtbiharmonic",help="Preprocessing of orig MRSI files (default: filtbiharmonic)")
parser.add_argument('--h', type=str, default="both",help="Display only one [lh,rh] hemisphere or both (default: both)")


args        = parser.parse_args()
parc_scheme = args.parc 
group       = args.group
dimalg      = args.dimalg
npert       = args.npert
perplexity  = args.perplexity
scale       = args.scale
preproc_str = args.preproc.replace("filt","")
dimalg      = args.dimalg
diag        = args.diag
hemisphere  = args.h

 
if dimalg=="pca_tsne":
    dimalg_str     = f"{dimalg}{perplexity}"
elif dimalg=="umap":
    dimalg_str     = f"{dimalg}"


# LOAD MSI 3D map
gm_mask       = datasets.load_mni152_gm_mask().get_fdata().astype(bool)
msiFileName   = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_str}_{dimalg_str}_desc-{diag}_metabsim3D_mrsi.nii.gz"
msiDirPath    = join(dutils.BIDSDATAPATH,group,"derivatives","group","msi","mrsi")     
msi_map_3D_np = nib.load(join(msiDirPath,msiFileName)).get_fdata().astype(int)

msiFileName   = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_str}_{dimalg_str}_desc-{diag}_metabsim1D_mrsi.npz"
msi_map_1D_np = np.load(join(msiDirPath,msiFileName))["msi_list"]
# LOAD MSI meta data
msi_data       = np.load(join(msiDirPath,msiFileName).replace("3D_mrsi.nii.gz","1D_mrsi.npz"))
parcel_labels  = msi_data["parcel_labels"]
parcel_names   = msi_data["parcel_names"]
super_regions  = np.array([region[0:9] for region in parcel_names])
super_regions  = np.array([region.replace("-lh","") for region in super_regions])
super_regions  = np.array([region.replace("-rh","") for region in super_regions])

# parcel image:
parcel_mni_img_nii          = nib.load(join(dutils.DEVDATAPATH,"atlas",
                                            f"chimera-{parc_scheme}-{scale}",
                                            f"chimera-{parc_scheme}-{scale}.nii.gz"))
parcel_mni_img_np           = parcel_mni_img_nii.get_fdata().astype(int)

# Load MeSiM data
conDirPath  = join(dutils.BIDSDATAPATH,group,"derivatives","group","connectivity","mrsi")
conFileName = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_str}_desc-{diag}_connectivity_mrsi.npz"
con_data                 = np.load(join(conDirPath,conFileName))
MeSiM_pop_avg            = con_data["MeSiM_pop_avg"]





###########################################################################


spectrum_colors_trans = colorbar.load_fsl_cmap("spectrum_iso_transparent",plotly=True)
mapper = umap.UMAP().fit(MeSiM_pop_avg)


hover_data = pd.DataFrame({'label':parcel_labels,
                           'names':parcel_names,
                           'msi':msi_map_1D_np})


con_output = umap_con.connectivity(mapper,
                               edge_bundling="hammer",
                               tsne_embedding=msi_map_1D_np.reshape(-1, 1),
                               use_dask=True,
                               iterations=1000,
                               node_names=parcel_names,
                               hemisphere=hemisphere,
                               )

node_names = con_output["node_names"]

fig   = umap_con.plot_connectivty(con_output,edge_color="black",
                                  node_cmap=spectrum_colors_trans,
                                  node_size=15,background="white")
fig.show()





    
