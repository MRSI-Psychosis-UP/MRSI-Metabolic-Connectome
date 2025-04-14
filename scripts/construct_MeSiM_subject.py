import os, sys, copy,shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from graphplot.simmatrix import SimMatrixPlot
import pandas as pd
import nibabel as nib
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
from connectomics.parcellate import Parcellate
from randomize.randomize import Randomize
from connectomics.network import NetBasedAnalysis
from connectomics.nettools import NetTools
from connectomics.mesim import MeSiM
from registration.registration import Registration
from randomize.randomize import Randomize
from tools.mridata import MRIData
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

METABOLITES           = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
PARC_CEREB_SCHEME     = "cerebellum" 
dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
parc     = Parcellate()
simplt   = SimMatrixPlot()
nba      = NetBasedAnalysis()
nettools = NetTools()
mesim    = MeSiM()
###############################################################################
FONTSIZE       = 16



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--parc', type=str, default="LFMIHIFIS", choices=['LFMIHIFIS', 'LFMIHIFIF'], 
                        help='Chimera parcellation scheme, choice must be one of: LFMIHIFIS [default], LFMIHIFIF')
    parser.add_argument('--scale',type=int,default=3,help="Cortical parcellation scale (default: 3)")
    parser.add_argument('--nthreads',type=int,default=4, help="Number of parallel threads (default=4)")
    parser.add_argument('--group', type=str, default='Dummy-Project', help='Group name (default: "Dummy-Project")')
    parser.add_argument('--npert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--subject_id', type=str, help='subject id', default="S001")
    parser.add_argument('--session', type=str, help='recording session',choices=['V1', 'V2', 'V3','V4','V5'], default="V1")
    parser.add_argument('--overwrite',type=int,default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--leave_one_out',type=int,default=0, choices = [1,0],help="Leave-one-metaobolite-out (default: 0)")
    parser.add_argument('--show_plot',type=int,default=0, choices = [1,0],help="Display similarity matrix plot (default: 0)")
    parser.add_argument('--preproc', type=str, default="filtbiharmonic",help="Preprocessing of orig MRSI files (default: filtbiharmonic)")
    parser.add_argument('--t1mask' , type=str, default=None,help="Anatomical T1w brain mask path")

    


    # Parse the arguments
    args = parser.parse_args()

    # Print the values for demonstration purposes
    debug.info(f"GM Parcellation: {args.parc}")
    debug.info(f"Group: {args.group}")
    debug.info(f"Number of perturbations: {args.npert}")
    debug.info(f"subject id: {args.subject_id}")
    debug.info(f"session: {args.session}")

    subject_id     = args.subject_id
    session        = args.session
    GROUP          = args.group
    ftools         = FileTools()
    npert          = args.npert
    OVERWRITE      = args.overwrite
    LEAVE_ONE_OUT  = bool(args.leave_one_out)
    NPROC          = args.nthreads
    SHOW_PLOT      = bool(args.show_plot)
    preproc_string = args.preproc
    scale          = args.scale
    parc_scheme    = args.parc
    t1mask_path_arg = args.t1mask

    ###############################################################################
    ############ Parcel List + Merge ##################
    sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","cer-rh",
                        "ctx-lh","subc-lh","thal-lh","cer-lh",]

    ################################################################################
    ############Construct Subject ID ##################
    prefix = f"sub-{subject_id}_ses-{session}"
    mridata             = MRIData(subject_id,session,group=GROUP)
    outfilepath         = mridata.get_connectivity_path("mrsi",parc_scheme,scale,npert)
    connectome_dir_path = split(outfilepath)[0]
    #
    outDirfigure_path = join(dutils.ANARESULTSPATH,"MeSiMs",GROUP,prefix)
    if exists(outfilepath):
        debug.success(prefix,"Already processed")
        if OVERWRITE:
            debug.warning(prefix,"Overwriting existing")
        else:
            return
    # MRSI Data
    mrsi_ref_img_path = mridata.get_mri_filepath(modality="mrsi",space="orig",
                                                 desc="signal",met="Ins",option=preproc_string)
    if not exists(mrsi_ref_img_path):
        debug.error("No MRSI data found")
        return

    debug.title(f"Compute Metabolic Simmilarity {prefix}")
    ##############################################################################
    ############## Get Parcel image in MRSI space #############
    ##############################################################################
    mrsi_orig_mask_nifti = mridata.get_mri_nifti(modality="mrsi",space="orig",desc="brainmask")
    mrsi_orig_mask_np    = mrsi_orig_mask_nifti.get_fdata().squeeze()

    parcel_mrsi_ni,path = mridata.get_parcel("mrsi",parc_scheme,scale)
    parcel_mrsi_np      = parcel_mrsi_ni.get_fdata()
    parcel_mrsi_header  = parcel_mrsi_ni.header
    parcel_header_dict  = parc.get_parcel_header(path.replace(".nii.gz",".tsv"))

    ############ Get parcels and mask outside MRSI region   #############
    if t1mask_path_arg:
        t1mask_orig_nifti = nib.load(t1mask_path_arg)
    else:
        t1mask_orig_nifti = mridata.get_mri_nifti(modality="t1w",space="orig",desc="brain")

    t1mask_orig_nifti  = mridata.get_mri_nifti(modality="t1w",space="orig",desc="brainmask")
    transform_list     = mridata.get_transform("inverse","mrsi")
    t1mask_mrsi_img    = reg.transform(mrsi_ref_img_path,t1mask_orig_nifti,transform_list).numpy()
    parcel_header_dict = parc.count_voxels_per_parcel(parcel_mrsi_np,mrsi_orig_mask_np,
                                                                    t1mask_mrsi_img,parcel_header_dict)
    # Extracting all label values without filtering on 'mask'
    all_labels_list         = [sub_dict['label'] for sub_dict in parcel_header_dict.values()]
    voxels_outside_mrsi     = {k: v for k, v in parcel_header_dict.items() if v['count'][-1] <= 5}
    # Extracting all 'label' values into a single list
    parcel_labels_ignore    = [sub_dict['label'] for sub_dict in voxels_outside_mrsi.values()]
    parcel_label_ids_ignore = [keys for keys in voxels_outside_mrsi.keys()]
    label_list_concat       = ["-".join(sublist) for sublist in all_labels_list]
    parcel_labels_ignore_concat = ["-".join(sublist) for sublist in parcel_labels_ignore]
    n_parcels               = len(parcel_header_dict)
    os.makedirs(connectome_dir_path,exist_ok=True)
    ############ Compute MeSiM   #############
    mrsirand       = Randomize(mridata,space="orig",option=preproc_string)
    simmatrix_sp, pvalue_sp,parcel_concentrations   = mesim.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,
                                                                            parcel_label_ids_ignore,npert,
                                                                            corr_mode = "spearman",
                                                                            rescale   = "zscore",n_proc=NPROC)
    simmatrix_sp_leave_out = None
    if LEAVE_ONE_OUT:
        simmatrix_sp_leave_out      = mesim.leave_one_out(simmatrix_sp,mrsirand,
                                                        parcel_mrsi_np,parcel_header_dict,
                                                        parcel_label_ids_ignore,npert,
                                                        corr_mode = "spearman",rescale="zscore")

    del parcel_header_dict[0] # Remove background
    labels_indices          = np.array(list(parcel_header_dict.keys()))
    np.trim_zeros(labels_indices)
    simmatrix_ids_to_delete = list()
    parcel_ids_positions, label_list_concat = parc.get_main_parcel_plot_positions(sel_parcel_list,label_list_concat)

    for idx_to_del in parcel_label_ids_ignore:
        simmatrix_ids_to_delete.append(np.where(labels_indices==idx_to_del)[0][0])

    ######### Save Results ########
    os.makedirs(connectome_dir_path,exist_ok=True)
    np.savez(f"{outfilepath}",
            parcel_concentrations   = parcel_concentrations,
            simmatrix_sp            = simmatrix_sp,
            pvalue_sp               = pvalue_sp,
            simmatrix_sp_leave_out  = simmatrix_sp_leave_out,
            labels                  = label_list_concat,
            labels_indices          = labels_indices,
            parcel_labels_ignore    = parcel_labels_ignore_concat,
            simmatrix_ids_to_delete = simmatrix_ids_to_delete,
            metabolites_leaveout    = METABOLITES)

    ftools.save_dict(parcel_header_dict,outfilepath.replace(".npz",".json"))
    debug.success(f"Results Saved to {outfilepath}")
    debug.separator()

    ################ Create Parcel distance matrix 
    parcel_mrsi     = ftools.numpy_to_nifti(parcel_mrsi_np.astype(int),parcel_mrsi_header)
    centroids_dict  = nettools.compute_parcel_centers(parcel_mrsi)
    centroids_arr   = [centroids_dict[parcel_id] for parcel_id in parcel_header_dict.keys() if parcel_id not in parcel_label_ids_ignore]
    distance_matrix = nettools.compute_distance_matrix(np.array(centroids_arr))
    try:    
        alpha    = 0.05/npert
        simmatrix_sp_corr = copy.deepcopy(simmatrix_sp)
        simmatrix_sp_corr[pvalue_sp>alpha] = 0
        fig, axs = plt.subplots(1,2, figsize=(16, 12))  # Adjust size as necessary
        plot_outpath = outfilepath.replace(".npz","_simmatrix")
        os.makedirs(outDirfigure_path,exist_ok=True)
        plot_outpath = join(outDirfigure_path,split(plot_outpath)[1])
        simplt.plot_simmatrix(simmatrix_sp,ax=axs[0],titles=f"{prefix} MeSiM",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,
                            colormap="bluewhitered",show_parcels="H") 
        simplt.plot_simmatrix(simmatrix_sp_corr,ax=axs[1],titles=f"{prefix} MeSiM corrected p>{alpha}",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,
                            colormap="bluewhitered",show_parcels="H",result_path = plot_outpath) 
        if SHOW_PLOT:
            plt.show()
        ######### Adjacency Matrix ########
        debug.info("Create Rich-Club Results")
        # Adj Matrix
        simmatrix_adjusted = copy.deepcopy(simmatrix_sp)
        # Delete specified rows & columns
        array_after_row_deletion = np.delete(simmatrix_adjusted, simmatrix_ids_to_delete, axis=0)
        simmatrix_adjusted       = np.delete(array_after_row_deletion, simmatrix_ids_to_delete, axis=1)
        non_zero_indices         = np.where(simmatrix_adjusted.sum(axis=0) != 0)[0]
        simmatrix_adjusted       = simmatrix_adjusted[non_zero_indices[:, None], non_zero_indices]
        density                  = 0.18
        simmatrix_binarized = nba.binarize(simmatrix_adjusted,threshold=density,
                                            mode="abs",threshold_mode="density")
        ######### MeSiM-Distance Distribution ########
        iu = np.triu_indices_from(simmatrix_adjusted, k=1)
        simmatrix_upper = simmatrix_adjusted[iu]
        distance_upper = distance_matrix[iu]
        #
        mask_pos = simmatrix_upper > 0
        mask_neg = simmatrix_upper < 0
        #
        df_pos = pd.DataFrame({
            "Distance": distance_upper[mask_pos].flatten(),
            "Correlation": simmatrix_upper[mask_pos].flatten(),
            "Type": "Positive"
        })
        df_neg = pd.DataFrame({
            "Distance": distance_upper[mask_neg].flatten(),
            "Correlation": simmatrix_upper[mask_neg].flatten(),
            "Type": "Negative"
        })
        df = pd.concat([df_pos, df_neg])

        ######### RichClub ########
        RC_ALPHA = 0.05
        node_deg, rc_coef, rand_rc_params = nba.compute_richclub_stats(simmatrix_binarized,alpha=RC_ALPHA)
        median_rand_rc = rand_rc_params["median"]
        lower_rand_rc  = rand_rc_params["lower"]
        upper_rand_rc  = rand_rc_params["upper"]
        ######### SimMatrix + Binarized PLots ########
        # plot_outpath = outfilepath.replace(".npz","_plot_adjacency")
        plot_outpath = join(outDirfigure_path,f"{prefix}_npert-{npert}_RichClub.pdf")
        fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
        simplt.plot_simmatrix(simmatrix_adjusted,ax=axs[0,0],titles=f"Weighted MeSiM",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,colormap="blueblackred") 
        # MeSiM vs Distance
        kde_neg = sns.kdeplot(data=df[df["Type"] == "Negative"],x="Distance",y="Correlation",
            fill=True,cmap="Blues",alpha=1, cbar=True,cbar_kws={"label": "Negative MeSiM KDE"},
            ax=axs[1,0])
        kde_pos = sns.kdeplot(data=df[df["Type"] == "Positive"],x="Distance",y="Correlation",
            fill=True,cmap="Reds",alpha=1, cbar=True,cbar_kws={"label": "Positive MeSiM KDE"},
            ax=axs[1,0])
        axs[1,0].grid()
        axs[1,0].set_xlabel('Distance',fontsize=FONTSIZE)
        axs[1,0].set_ylabel('MeSiM',fontsize=FONTSIZE)
        # RC plot
        axs[1,1].fill_between(node_deg, lower_rand_rc, upper_rand_rc, color='gray', alpha=0.5, label=f"Null Distribution (α={RC_ALPHA})")
        axs[1,1].plot(node_deg, median_rand_rc,'--', color='black')
        axs[1,1].plot(node_deg, rc_coef, label='Metabolic Network', color='green')
        axs[1,1].set_xlabel('Degree',fontsize=FONTSIZE)
        axs[1,1].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE)
        axs[1,1].legend()
        axs[1,1].grid()

        simmatrix_binarized_pn = nba.binarize(simmatrix_adjusted,threshold=density,
                                    mode="posneg",threshold_mode="density")
        simplt.plot_simmatrix(simmatrix_binarized_pn,ax=axs[0,1],titles=f"Binarized MeSiM [desnity={density}]",
                            parcel_ids_positions=parcel_ids_positions,colormap="blueblackred",
                            scale_factor=0.6,
                            result_path = plot_outpath)
        plt.tight_layout() 
        if SHOW_PLOT:
            plt.show()

    except Exception as e:
        debug.error("Failed creating results",e)
        return



if __name__ == "__main__":
    main()





    


