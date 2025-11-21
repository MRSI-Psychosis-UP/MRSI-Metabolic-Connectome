import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, image, datasets
from scipy import interpolate
import copy, sys
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from graphplot.colorbar import ColorBar
from scipy import interpolate, ndimage


import argparse
colorbar    = ColorBar()

dutils       = DataUtils()       
debug        = Debug()
FONTSIZE     = 16
confInterval = 0.01





def smooth_and_interpolate(x, y, window_size=5, interp_kind='linear', num_new_points=100):
    x = np.array(x)
    y = np.array(y)
    # Interpolation
    if num_new_points is not None:
        x_new = np.linspace(np.min(x), np.max(x), num_new_points)
    else:
        x_new = x
    mask = np.diff(y) != 0
    mask = np.insert(mask, 0, True)
    f = interpolate.interp1d(x[mask], y[mask], kind=interp_kind, fill_value="extrapolate")
    y_interpolated = f(x_new)
    # Boundary-aware smoothing
    y_smoothed = ndimage.uniform_filter1d(y_interpolated, size=window_size, mode='nearest')
    return x_new, y_smoothed


parser = argparse.ArgumentParser(description="Process some input parameters.")
parser.add_argument('--group', type=str,default="Geneva-Study") 
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
alpha       = 1-confInterval

 
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
metab_profiles_subj_list = con_data["metab_profiles_subj_list"]
metab_profiles_subj_avg  = metab_profiles_subj_list.mean(axis=0)
metabolites              = ["tNAA", "Ins", "Cho", "Glx", "tCr"]
n_met                    = len(metabolites) 


# Calculate histogram bins
_, msi_bins           = np.histogram(msi_map_1D_np.flatten(), bins=8)
metab_profile_bin_med = np.zeros((len(msi_bins) - 1, n_met))  # One profile per bin, with 5 features
metab_profile_bin_upp = np.zeros((len(msi_bins) - 1, n_met))
metab_profile_bin_low = np.zeros((len(msi_bins) - 1, n_met))
for bin_id in range(len(msi_bins) - 1):
    # Find indices within the current bin range
    parcel_index_list = np.where((msi_map_1D_np > msi_bins[bin_id]) & (msi_map_1D_np < msi_bins[bin_id + 1]))[0]
    # Initialize the metabolic profile for this bin
    metab_profile_arr = list()
    if len(parcel_index_list)==0:
        debug.info(parcel_index_list)
        continue
    metab_profile_arr = np.array([metab_profiles_subj_avg[i] for i in parcel_index_list])
    if len(parcel_index_list) > 0:
        metab_profile_bin_med[bin_id] = np.median(metab_profile_arr, axis=(0,2))
        #  
        metab_profile_bin_low[bin_id] = np.nanpercentile(metab_profile_arr, (alpha / 2) * 100, axis=(0,2))
        metab_profile_bin_upp[bin_id] = np.nanpercentile(metab_profile_arr, (1 - alpha / 2) * 100, axis=(0,2))
    else:
        metab_profile_bin_med[bin_id] = np.zeros(5)  # Or keep as zero profile if no points in bin
    

############################################################3
# Set up figure and axes with gridspec
fig           = plt.figure(figsize=(8.27, 5), constrained_layout=True)
gs            = fig.add_gridspec(nrows=2, height_ratios=[0.2, 1])
spectrum_cmap = colorbar.load_fsl_cmap("spectrum_iso", plotly=False)

# Create the horizontal colorbar
cbar_ax = fig.add_subplot(gs[0])
gradient = np.linspace(0, 1, 256).reshape(1, -1)
cbar_ax.imshow(gradient, aspect='auto', cmap=spectrum_cmap)
cbar_ax.set_axis_off()  # Hide axes for colorbar

# Main XY plot
ax = fig.add_subplot(gs[1])

# Your plotting logic here
x_arr, y_arr           = [], []
metab_profile_extended = []
homotopy_bins_centroid = np.linspace(0, 255, len(msi_bins) - 1)

for idm, metabolite in enumerate(metabolites):
    _x, _y = homotopy_bins_centroid, metab_profile_bin_med[:, idm]
    x, y = smooth_and_interpolate(_x, _y, window_size=10, interp_kind='cubic', num_new_points=100)
    xu, upper = smooth_and_interpolate(_x, metab_profile_bin_upp[:, idm], window_size=10, interp_kind='cubic', num_new_points=100)
    xl, lower = smooth_and_interpolate(_x, metab_profile_bin_low[:, idm], window_size=10, interp_kind='cubic', num_new_points=100)
    ax.plot(x, y, label=metabolite, linewidth=1.5)
    ax.fill_between(xu, lower, upper, alpha=0.23)
    metab_profile_extended.append(y)
    x_arr.append(x)
    y_arr.append(y)

metab_profile_extended = np.array(metab_profile_extended)
homotopic_index_extended = x

# Set labels and style
ax.set_xlabel("Metabolic Similarity Index [AU]", fontsize=FONTSIZE)
ax.set_ylabel("Z-score", fontsize=FONTSIZE)
ax.set_xlim(0, max(x))
# ax.set_title(f"MSI {group} chimera{parc_scheme}-{scale} {dimalg_str} ")
ax.grid(True)
ax.tick_params(axis='y', labelsize=FONTSIZE - 2)
ax.legend(fontsize=FONTSIZE - 4,title=f"CI={confInterval}")

# Save and display the plot
plt.tight_layout()
msiFileName = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_str}_{dimalg_str}_desc-{diag}_msi_inverse_scale_mrsi.nii.gz"
result_path = join(msiDirPath, msiFileName)
plt.savefig(f"{result_path}.pdf")
np.savez(result_path.replace(".pdf", "_raw.npz"),
         x_arr=np.array(x_arr),
         y_arr=np.array(y_arr),
         metabolites=metabolites)

plt.show()