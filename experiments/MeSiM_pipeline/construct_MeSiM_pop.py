import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split, exists
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from tools.mridata import MRIData
import nibabel as nib
from connectomics.netcluster import NetCluster
import copy, sys
import pandas as pd
from connectomics.network import NetBasedAnalysis
from connectomics.nettools import NetTools
from connectomics.parcellate import Parcellate
from connectomics.simmilarity import Simmilarity
import argparse
from rich.progress import track
import seaborn as sns

dutils    = DataUtils()
simm      = Simmilarity()
debug     = Debug()
netclust  = NetCluster()
parc      = Parcellate()
ftools    = FileTools()
simplt    = SimMatrixPlot()
nettools  = NetTools()
nba       = NetBasedAnalysis()
FONTSIZE  = 16


def main():
# Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--parc', type=str, default="LFMIHIFIS", choices=['LFMIHIFIS', 'LFMIHIFIF','LFIIIIFIS'], 
                        help='Chimera parcellation scheme, choice must be one of: LFMIHIFIS [default], LFMIHIFIF')
    parser.add_argument('--scale',type=int,default=3,help="Cortical parcellation scale (default: 3)")
    parser.add_argument('--group', type=str, default='Dummy-Project', help='group name (default: "Mindfulness-Project")')
    parser.add_argument('--npert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--overwrite',type=int,default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--preproc', type=str, default="filtbiharmonic",help="Preprocessing of orig MRSI files (default: filtbiharmonic)")
    parser.add_argument('--analyze',type=int,default=0, choices = [1,0],help="Apply RichClub Analysis (default: 0)")
    parser.add_argument('--participants', type=str, default=None,
                    help="Path to TSV file containing list of participant IDs and sessions to include. If not specified, process all.")
    parser.add_argument('--alpha', type=float, default=0.05, help='MeSiM corr correction before Bonferroni (default: 0.05)')
    parser.add_argument('--results_dir_path' , type=str, default=None,help=f"Directory path where results figures will be saved (default: {dutils.ANARESULTSPATH})")

    ################################################################################
    args        = parser.parse_args()
    parc_scheme        = args.parc 
    scale       = args.scale 
    group       = args.group
    npert       = args.npert
    preproc_str = args.preproc.replace("filt","")
    overwrite   = bool(args.overwrite)
    analyze     = bool(args.analyze)
    participants_file = args.participants
    alpha_bf    = args.alpha/args.npert
    outDirfigure_path = args.results_dir_path

    ################################################################################


    # participants_file = "experiments/MeSiM_pipeline/participant_list/best_Mindfulness-Project_sessions_mrsi.tsv"
    if participants_file is None:
        participant_session_list = join(dutils.BIDSDATAPATH,group,"participants_allsessions.tsv")
        df                       = pd.read_csv(participant_session_list, sep='\t')
        df = df[df.session_id != "V2BIS"]
    else:
        df = pd.read_csv(participants_file, sep='\t')

    subject_id_list = df.participant_id.to_list()
    session_id_list = df.session_id.to_list()

    debug.title(f"Compute population MeSiM for {group} and atlas {parc_scheme} scale-{scale} - npert {npert}")



    ####################################

    MeSiM_subjects_list       = list()
    metab_profiles_subjects = list()

    for subject_id,session in track(zip(subject_id_list,session_id_list), 
                                total=len(subject_id_list), 
                                description="Extracting MeSiMs..."):
        
        prefix   = f"sub-{subject_id}_ses-{session}"
        mridata  = MRIData(subject_id,session,group=group)
        try:
            con_data   = np.load(mridata.get_connectivity_path("mrsi",parc_scheme,scale,npert,
                                                            filtoption=preproc_str))
            sim_matrix = con_data["simmatrix_sp"]
            p_values   = con_data["pvalue_sp"]
            sim_matrix[p_values>alpha_bf] = 0
            # 
            metab_profiles_subjects.append(con_data["parcel_concentrations"])
            MeSiM_subjects_list.append(sim_matrix)
        except Exception as e:
            debug.warning(prefix,e)

    MeSiM_subjects_list     = np.array(MeSiM_subjects_list)
    metab_profiles_subjects = np.array(metab_profiles_subjects)

    # Load parcelation
    # Group
    parcel_labels_group = copy.deepcopy(con_data["labels_indices"])
    _str_arr            = copy.deepcopy(con_data["labels"])
    parcel_names_group  = np.array([s for s in _str_arr if s != "BND"])


    debug.info(f"Collected {MeSiM_subjects_list.shape[0]} MeSiMs of shape {MeSiM_subjects_list.shape[1::]}")

    ########## Clean simmilarity matrices ##########

    # Diascard sparse subjects MeSiM from average 
    debug.title("Exclude sparse within subject-wise MeSiMs")
    MeSiM_list_sel,i,e = simm.filter_sparse_matrices(MeSiM_subjects_list,sigma=5)
    metab_profiles_subjects_sel = np.delete(metab_profiles_subjects,e,axis=0)
    session_id_arr_sel      = np.delete(session_id_list,e,axis=0)
    subject_id_arr_sel      = np.delete(subject_id_list,e,axis=0)
    MeSiM_list_sel          = np.array(MeSiM_list_sel)
    discarded_subjects      = [subject_id_list[idx] for idx in np.array(e)]
    discarded_sessions      = [session_id_list[idx] for idx in np.array(e)]
    debug.info(f"Excluded {len(e)} sparse MeSiMs of shape, remaining {MeSiM_list_sel.shape[0]}")


    ############# Detect empty correlations from pop AVG  #############
    MeSiM_pop_avg             = MeSiM_list_sel.mean(axis=0)
    # Cleanup empty nodes
    mask_parcel_indices         = np.where(np.diag(MeSiM_pop_avg) == 0)[0]
    debug.title("Remove sparse nodes")
    # delete rowd/cols of empty correlations 
    _MeSiM_pop_avg_clean          = np.delete(MeSiM_pop_avg, mask_parcel_indices, axis=0)
    MeSiM_pop_avg_clean           = np.delete(_MeSiM_pop_avg_clean, mask_parcel_indices, axis=1)
    metab_profiles_subjects_clean = np.delete(metab_profiles_subjects_sel, mask_parcel_indices, axis=1)
    _MeSiM_subjects_list          = np.delete(MeSiM_subjects_list, mask_parcel_indices, axis=1)
    MeSiM_subjects_list_clean     = np.delete(_MeSiM_subjects_list, mask_parcel_indices, axis=2)


    for i in mask_parcel_indices:
        debug.info("Removed node",parcel_labels_group[i],parcel_names_group[i])

    debug.separator()
    debug.info(f"Final MeSiM shape {MeSiM_pop_avg_clean.shape}")
    # same for parcellation data 
    parcel_labels_group  = np.delete(parcel_labels_group, mask_parcel_indices)
    parcel_names_group   = np.delete(parcel_names_group, mask_parcel_indices)
    n_parcels_group      = len(parcel_names_group)


    ############## Save intermdiate simmatrices and parcel conc


    resultssubdir = join(dutils.BIDSDATAPATH,group,"derivatives","group","connectivity","mrsi")
    _outpath = mridata.get_connectivity_path("mrsi",parc_scheme,scale,npert,
                                                            filtoption=preproc_str)
    filename = split(_outpath)[1].replace(f"sub-{subject_id}_ses-{session}",group)
    filename = filename.replace("connectivity","group_connectivity")
    MeSiM_outpath  = join(resultssubdir,filename)


    np.savez(MeSiM_outpath,
                    metab_profiles_subj_list = metab_profiles_subjects_sel,
                    MeSiM_subj_list          = MeSiM_subjects_list_clean,
                    MeSiM_pop_avg            = MeSiM_pop_avg_clean,
                    parcel_labels_group      = parcel_labels_group,
                    parcel_names_group       = parcel_names_group,
                    subject_id_list          = subject_id_arr_sel,
                    session_id_list          = session_id_arr_sel,
                    discarded_subjects       = discarded_subjects,
                    discarded_sessions       = discarded_sessions)

    debug.success("Saved cleaned subject MeSiMs and population averaged MeSiM to \n",MeSiM_outpath)




    if not analyze:return 
    ################ Create Parcel distance matrix 
    _parc_scheme       = "LFMIHIFIF"
    parcel_path        = join(dutils.DEVDATAPATH,"atlas",f"chimera-{_parc_scheme}-{scale}",
                                        f"chimera-{_parc_scheme}-{scale}.nii.gz")
    parcel_mrsi         = nib.load(parcel_path)
    parcel_mrsi_np      = parcel_mrsi.get_fdata()
    parcel_mrsi_header  = parcel_mrsi.header

    # Filter out exlcuded labels
    for label in np.unique(parcel_mrsi_np).astype(int):
        if label not in parcel_labels_group:
            debug.info("Remove",label)
            parcel_mrsi_np[parcel_mrsi_np == label] = 0

    parcel_mrsi     = ftools.numpy_to_nifti(parcel_mrsi_np,parcel_mrsi_header)
    centroids_dict  = nettools.compute_parcel_centers(parcel_mrsi)
    centroids_arr   = np.array(list(centroids_dict.values()))
    # centroids_arr   = [centroids_dict[parcel_id] for parcel_id in parcel_labels_group]

    distance_matrix = nettools.compute_distance_matrix(np.array(centroids_arr))
    try:
        fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
        results_filename = MeSiM_outpath.replace("group_connectivity_mrsi.npz","richclub.pdf")
        if outDirfigure_path is None:
            outDirfigure_path = join(dutils.ANARESULTSPATH,"MeSiMs",group)
            debug.info("Setting result figures output directory to ",outDirfigure_path)    
        

        ######### Adjacency Matrix ########
        debug.info("Create Rich-Club Results")
        # Adj Matrix

        density             = 0.18
        simmatrix_binarized = nba.binarize(MeSiM_pop_avg_clean,threshold=density,
                                            mode="abs",threshold_mode="density")
        ######### MeSiM-Distance Distribution ########
        iu = np.triu_indices_from(MeSiM_pop_avg_clean, k=1)
        simmatrix_upper = MeSiM_pop_avg_clean[iu]
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
        np.fill_diagonal(simmatrix_binarized,0)
        node_deg, rc_coef, rand_rc_params = nba.compute_richclub_stats(simmatrix_binarized,alpha=RC_ALPHA)
        median_rand_rc = rand_rc_params["median"]
        lower_rand_rc  = rand_rc_params["lower"]
        upper_rand_rc  = rand_rc_params["upper"]
        ######### SimMatrix + Binarized PLots ########
        
        simplt.plot_simmatrix(MeSiM_pop_avg_clean,ax=axs[0,0],titles=f"MeSiM [removed empty nodes]",
                            scale_factor=0.4,
                            colormap="blueblackred") 
        simmatrix_binarized_pn = nba.binarize(MeSiM_pop_avg_clean,threshold=density,
                                    mode="posneg",threshold_mode="density")
        simplt.plot_simmatrix(simmatrix_binarized_pn,ax=axs[0,1],titles=f"Binarized MeSiM [density={density}]",
                            colormap="blueblackred",
                            scale_factor=0.6)
        # MeSiM vs Distance
        kde_neg = sns.kdeplot(data=df[df["Type"] == "Negative"],x="Distance",y="Correlation",
            fill=True,cmap="Blues",alpha=1, cbar=True,cbar_kws={"label": "Negative MeSiM KDE"},
            ax=axs[1,0])
        kde_pos = sns.kdeplot(data=df[df["Type"] == "Positive"],x="Distance",y="Correlation",
            fill=True,cmap="Reds",alpha=1, cbar=True,cbar_kws={"label": "Positive MeSiM KDE"},
            ax=axs[1,0])
        axs[1,0].grid()
        axs[1,0].set_xlabel('Distance',fontsize=FONTSIZE)
        axs[1,0].set_ylabel('Correlations',fontsize=FONTSIZE)
        # RC plot
        axs[1,1].fill_between(node_deg, lower_rand_rc, upper_rand_rc, color='gray', alpha=0.5, label=f"Null Distribution (α={RC_ALPHA})")
        axs[1,1].plot(node_deg, median_rand_rc,'--', color='black')
        axs[1,1].plot(node_deg, rc_coef, label='Metabolic Network', color='green')
        axs[1,1].set_xlabel('Degree',fontsize=FONTSIZE)
        axs[1,1].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE)
        axs[1,1].legend()
        axs[1,1].grid()
        ##########################################################
        plot_outpath = join(outDirfigure_path,results_filename)
        fig.savefig(plot_outpath)
        plt.tight_layout() 
        plt.show()

    except Exception as e:
        debug.error("Failed creating results",e)
        return



if __name__ == "__main__":
    main()
