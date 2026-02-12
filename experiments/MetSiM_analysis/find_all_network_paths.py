import numpy as np
import os, sys, copy, csv, time
from os.path import join, exists, split
import pandas as pd
from nilearn import datasets
import nibabel as nib
import argparse
from dipy.viz import window
from mrsitoolbox.graphplot.netplot import NetPlot


from mrsitoolbox.tools.datautils import DataUtils
from mrsitoolbox.tools.debug import Debug
from mrsitoolbox.connectomics.parcellate import Parcellate
from mrsitoolbox.connectomics.network import NetBasedAnalysis
from mrsitoolbox.connectomics.nettools import NetTools
from mrsitoolbox.connectomics.netfibre import NetFibre




##############################################
FONTSIZE               = 16
##############################################
################################################################################################
parser = argparse.ArgumentParser(description="Process some input parameters.")
parser.add_argument('--group', type=str,default="LPN-Project") 
parser.add_argument('--parc', type=str, default="LFMIHIFIS",
                    help='Chimera parcellation scheme, valid choices: LFMIHIFIS, LFMIHIFIF,LFIIIIFIS. Default: LFIIIIFIS')
parser.add_argument('--scale', type=int, default=3,
                    help="Cortical parcellation scale (default: 3)")
parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1],
                    help="Overwrite existing parcellation (default: 0)")
parser.add_argument('--lobe', type=str, default="subc",help="Cortical 'ctx' or subCortical 'subc' region ('ctx'[default])")
parser.add_argument('--hemi', type=str, default="rh",help="hemisphere ('rh'[default])")
parser.add_argument('--start', type=int,help="Start node label number")
parser.add_argument('--stop', type=int,help="End node label number")
parser.add_argument('--lpath', type=int,default=12,help="Path length cutoff for DFS path search")
parser.add_argument('--alpha' , type=float, default=0.65 ,
                            help="MRSI-to-T1 parcel coverage above which enough MRSI singal has been recorded (default: 0.65)")
args        = parser.parse_args()
parc_scheme = args.parc 
scale       = args.scale
start_node  = args.start
stop_node   = args.stop
path_length = args.lpath
brainlobe   = args.lobe
hemisphere  = args.hemi
group       = args.group
mrsi_cov        = args.alpha
################################################################################################
hemisphere_str = "Left" if hemisphere=="lh" else "Right"
sym_hemisphere = "lh" if hemisphere=="rh" else "rh"
sym_hemisphere_str = "Left" if sym_hemisphere=="lh" else "Right"
lobe_other     = ["subc","thal","amygd","hypo","hipp","cer","brain-stem"] if brainlobe=="ctx" else ["ctx"]
################################################################################################
if start_node is None or stop_node is None: 
    if parc_scheme=="LFMIHIFIS":
        if brainlobe=="ctx":
            if hemisphere=="rh":
                start_node = 44
                stop_node  = 40
            elif hemisphere=="lh":
                start_node = 182
                stop_node  = 178
        elif brainlobe=="subc":
            if hemisphere=="rh":
                start_node = 109
                stop_node  = 2007
                path_length = 9
            elif hemisphere=="lh":
                start_node = 247
                stop_node  = 2005  
                path_length = 9

debug        = Debug()
parc         = Parcellate()
nettools     = NetTools()
dutils       = DataUtils()
nba          = NetBasedAnalysis()
mni_template = datasets.load_mni152_template()


debug.title(f"Construct all network path for atlas {parc_scheme}-{scale} in brainlob {hemisphere}-{brainlobe} from {start_node} to {stop_node}")
##############################################
atlas = f"cubic-{scale}" if "cubic" in parc_scheme else f"chimera-{parc_scheme}-{scale}" 
resultdir = join(dutils.BIDSDATAPATH,group,"derivatives","group","networkpaths",atlas)
os.makedirs(resultdir,exist_ok=True)

########################### parcel image: ###########################
gm_mask            = datasets.load_mni152_gm_mask().get_fdata().astype(bool)
parcel_image_path  = join(dutils.DEVDATAPATH,"atlas",
                                            f"chimera-{parc_scheme}-{scale}",
                                            f"chimera-{parc_scheme}-{scale}.nii.gz")
                                            
parcel_mni_img_nii          = nib.load(parcel_image_path)
parcel_mni_img_np           = parcel_mni_img_nii.get_fdata().astype(int)

ignore_parcel_list = list()
ignore_parcel_list.extend(lobe_other)
ignore_parcel_list.extend([f"{sym_hemisphere}",f"{sym_hemisphere_str}"])
parcel_labels,parcel_names,_ = parc.read_tsv_file(parcel_image_path.replace("nii.gz","tsv")
                                                  ,ignore_parcel_list=ignore_parcel_list)


################ Ignore Parcels defined by low MRSI coveraged <-> QMASK ####################
qmask_dir      = join(dutils.BIDSDATAPATH,group,"derivatives","group","qmask")
qmask_path     = join(qmask_dir, f"{group}_space-mni_met-CrPCr_desc-qmask_mrsi.nii.gz")
qmask_pop      = nib.load(qmask_path)
n_voxel_counts_dict = parc.count_voxels_inside_parcel(qmask_pop.get_fdata(), 
                                                        parcel_mni_img_np, 
                                                        parcel_labels)
ignore_parcel_idx = [index for index in n_voxel_counts_dict if n_voxel_counts_dict[index] < mrsi_cov]
ignore_rows = [np.where(parcel_labels == parcel_idx)[0][0] for parcel_idx in ignore_parcel_idx if len(np.where(parcel_labels == parcel_idx)[0]) != 0]
ignore_rows = np.sort(np.array(ignore_rows)) 

# Delete nodes in MeSiM
if len(ignore_rows)!=0:
    parcel_labels           = np.delete(parcel_labels,ignore_rows)
    parcel_names            = np.delete(parcel_names,ignore_rows)
else:
    debug.info("No low coverage parcels detected")


# Construct GM adjacency
centroids         = nettools.compute_centroids(parcel_mni_img_nii,parcel_labels,world=True)
centroids_mni     = nettools.compute_centroids(parcel_mni_img_nii,parcel_labels,world=False)

centroid_dict     = dict()
for i,image_label in enumerate(parcel_labels):
    centroid_dict[image_label]  = centroids[i]

netfibre     = NetFibre(None,centroid_dict,None,None)

adjacency_dict    = nettools.compute_adjacency_list(parcel_mni_img_np,
                                                    labels       = parcel_labels,
                                                    parcel_names = parcel_names,
                                                    order        = 1,
                                                    return_matrix = False)




if "subc" in brainlobe:
    if parc_scheme=="LFMIHIFIS":
        try:
            if hemisphere=="rh":
                adjacency_dict[118].add(109)
                adjacency_dict[109].add(118)
                # Brainstem ↔ Cerebellar lobules (Right side + vermis)
                debug.info("Connectiong BS with RH-CER")
                for cer_label in [2002, 2004, 2007, 2010, 2013, 2016, 2019, 2022,
                                2006, 2009, 2012, 2015, 2018, 2021]:  # include vermis
                    for bs_label in [277, 278, 279]:
                        if bs_label in parcel_labels and cer_label in parcel_labels:
                            try:
                                adjacency_dict[cer_label].add(bs_label)
                                adjacency_dict[bs_label].add(cer_label)
                            except Exception as e:pass#skip if node not present in qmask atlas
            elif hemisphere=="lh":
                debug.info("Connectiong BS with LH-CER")
                adjacency_dict[251].add(248)
                adjacency_dict[248].add(251)
                # Brainstem ↔ Cerebellar lobules (Right side + vermis)
                for cer_label in [2001, 2003, 2005, 2008, 2011, 2014, 2017, 2020,
                                2006, 2009, 2012, 2015, 2018, 2021]:  # include vermis
                    for bs_label in [277, 278, 279]:
                        if bs_label in parcel_labels and cer_label in parcel_labels:
                            try:
                                adjacency_dict[cer_label].add(bs_label)
                                adjacency_dict[bs_label].add(cer_label)
                            except Exception as e:pass#skip if node not present in qmask atlas
        except Exception as f:pass#skip if node not present in qmask atlas




adjacency_mat = nettools.graphdict_to_mat(adjacency_dict)
filename = f"{group}_{atlas}_{hemisphere}_{brainlobe}_desc-GMadjacency"
outpath = join(resultdir,filename)
np.savez(outpath,
         adjacency_mat=adjacency_mat,
         parcel_labels = parcel_labels,
         parcel_names = parcel_names)


netplot      = NetPlot(window)

netplot.scene.background((1, 1, 1))  # Set background to white
netplot.add_brain( mni_template, hemisphere, parcel_mni_img_np, 
                  parcel_labels_list = None,
                  opacity = 0.1)

netplot.add_gm_adjacency(adjacency_mat, centroids_mni,
                         node_radius=1,edge_color=(0,0,0),
                         edge_opacity=1,node_labels=parcel_labels.astype(str))
window.show(netplot.scene)


# len(paths_noise_ctx)
########## Find all possible path via DFS ##########
debug.info("Find all possible paths via DFS")
t0 = time.time()
paths = netfibre.find_all_simple_paths(adjacency_dict,
                                           start_node,
                                           stop_node,
                                           cutoff=path_length)

t1 = time.time()
debug.success("Found", len(paths), "paths in", f"{t1 - t0:.3f}s")
length_arr = np.array([len(path) for path in paths])
# plt.hist(length_arr)
# plt.show()

filename = f"{group}_{atlas}_{hemisphere}_{brainlobe}_desc-start-{start_node}_stop-{stop_node}_l-{path_length}_netpaths"
outpath = join(resultdir,filename)

np.save(outpath, np.array(paths, dtype=object))
debug.success("Saved paths to file",outpath)




