import os, sys, json
import numpy as np
from registration.registration import Registration
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
from os.path import join, split
from tools.mridata import MRIData
from connectomics.parcellate import Parcellate
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'
import nibabel as nib
from nilearn import datasets
import argparse, copy




parc     = Parcellate()
dutils   = DataUtils()
debug    = Debug()
reg      = Registration()



header_mni        = datasets.load_mni152_template().header

def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--atlas', type=str, default="LFMIHIFIF-3", 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4)')
    parser.add_argument('--overwrite',type=int,default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--group', type=str,default="Dummy-Project") 
    parser.add_argument('--subject_id', type=str, help='subject id', default="S001")
    parser.add_argument('--session', type=str, help='recording session',choices=['V1', 'V2', 'V3','V4','V5'], default="V1")
    parser.add_argument('--t1_pattern',type=str,default="_run-01_acq-memprage_",help="T1w file pattern e.g _run-01_acq-memprage_")

    args       = parser.parse_args()
    ATLAS_TYPE = args.atlas
    GROUP      = args.group
    OVERWRITE  = args.overwrite
    t1pattern  = args.t1_pattern
    ftools     = FileTools()


    ##### Set path ###########
    BIDS_ROOT_PATH     = join(dutils.BIDSDATAPATH,GROUP)
    PARC_PATH          = join(BIDS_ROOT_PATH,"derivatives","chimera-atlases")
    ##### Merge  LFMIHIFIF-3 parcels#####
    merge_parcels_dict = {}
    MERGE_PARCEL_PATH = join(dutils.DEVANALYSEPATH,"connectomics","data",f"merge_parcels_{ATLAS_TYPE}.json")
    if exists(MERGE_PARCEL_PATH):
        with open(MERGE_PARCEL_PATH, 'r') as file:
            merge_parcels_dict = json.load(file)
    ############ Set up Recording ID ##################
    subject_id, session = args.subject_id, args.session
    prefix                  = f"sub-{subject_id}_ses-{session}"
    outdir_path             = join(PARC_PATH,f"sub-{subject_id}",f"ses-{session}","anat")
    prefix_name             = f"{prefix}_run-01_acq-memprage_space-mni_atlas-{ATLAS_TYPE}-cer_dseg"
    parcimage_anat_outpath  = join(outdir_path,f"{prefix_name}.nii.gz")

    if exists(parcimage_anat_outpath):
        debug.success("Already parcellated",prefix)
        if not OVERWRITE:
            debug.warning("Exit")
            sys.exit()
        else: debug.warning("Overwriting existing",prefix)
    mridata = MRIData(subject_id,session,group=GROUP,t1_pattern=t1pattern)
    debug.separator()
    t1_path = mridata.data["t1w"]["brain"]["orig"]["path"]
    header_t1 = nib.load(t1_path).header
    t1_anat_np = nib.load(t1_path).get_fdata()
    if not exists(t1_path):debug.warning("SKIP",prefix);sys.exit()
    debug.title(f"Parcellating {prefix}")

    # Integrate Cerebellumj parcellation
    parcel_image_anat_gm_path = mridata.data["parcels"][ATLAS_TYPE]["orig"]["path"]
    debug.info("ATLAS_TYPE",ATLAS_TYPE)
    parcel_image_anat_nifti   = parc.get_parcellation_image(parcel_image_anat_gm_path,ignore_parcel_list=["cer-","wm-"])
    # Transform base parcellation to ANAT subject space 
    transform_list            = mridata.get_transform("forward","anat")
    parcel_image_mni_gm       = reg.transform(datasets.load_mni152_template(),parcel_image_anat_nifti,transform_list,
                                            interpolator_mode="genericLabel").numpy().astype(int)
    # Get parcel header and filter out main cerrebelum and cortical wm 
    parcel_header_path                               = mridata.data["parcels"][ATLAS_TYPE]["orig"]["labelpath"]
    indices_gm, labels_gm, _                         = parc.read_tsv_file(parcel_header_path,ignore_parcel_list=["cer-","wm-"])
    filtered_parcel_image = np.zeros(parcel_image_mni_gm.shape)
    for parcel_idx in indices_gm:
        filtered_parcel_image[parcel_image_mni_gm==parcel_idx] = parcel_idx
    parcel_image_mni_gm = copy.deepcopy(filtered_parcel_image)
    # Merge GM parcels in Lausanne atlas
    parcel_header_dict                      = parc.get_parcel_header(parcel_header_path,ignore_parcel_list=["cer-","wm-"])
    parcel_image_mni_gm ,parcel_header_dict = parc.merge_parcels(parcel_image_mni_gm,parcel_header_dict, merge_parcels_dict)
    indices_gm      = parcel_header_dict.keys()
    labels_gm       = [parcel_el["label"] for parcel_el in parcel_header_dict.values()]
    labels_gm       = ["-".join(sublist) for sublist in labels_gm]
    parcel_image_mni_cer, labels_cer, indices_cer, _ = parc.create_parcel_image(atlas_string="cerebellum")
    # ftools.save_nii_file(parcel_image_mni_gm,header_mni,"gm_parcellation_nifti.nii.gz")
    parcel_image_mni         = parc.merge_gm_wm_parcel(parcel_image_mni_gm, parcel_image_mni_cer).astype(int)
    labels_cer=[]

    # create gm, cer and wm indices and labels
    labels = list()
    labels.extend(labels_gm)
    labels.extend(labels_cer)
    indices = list()
    indices.extend(indices_gm)
    indices.extend(indices_cer)
    # remove background parcel
    for i,parcel_idx in enumerate(indices):
        if parcel_idx==0:break
    indices = np.delete(indices,i)
    labels  = np.delete(labels,i)

    ######## Transform and save in MNI; ANAT and MRSI space ########
    os.makedirs(outdir_path,exist_ok=True)
    # MNI space
    mni_atlas_outpath = join(outdir_path,f"{prefix_name}.nii.gz")
    debug.success("Saved parcel image in MNI space to",join(outdir_path,f"{prefix_name}.nii.gz"))
    ftools.save_nii_file(parcel_image_mni,header_mni,mni_atlas_outpath)
    parc.create_tsv(labels,indices,join(outdir_path,f"{prefix_name}.tsv"))

    # ANAT space
    header_t1         = nib.load(t1_path).header
    transform_list    = mridata.get_transform("inverse","anat")
    parcel_image_orig = reg.transform(t1_path,mni_atlas_outpath,transform_list,interpolator_mode="genericLabel")
    #
    prefix_name       = prefix_name.replace("mni","orig")
    anat_outpath      = join(outdir_path,f"{prefix_name}.nii.gz")
    debug.success("Saved parcel image in ANAT space to",anat_outpath)
    ftools.save_nii_file(parcel_image_orig.numpy(),header_t1,anat_outpath)
    parc.create_tsv(labels,indices,join(outdir_path,f"{prefix_name}.tsv"))

    # MRSI space
    mrsi_ref_img_path = mridata.data["mrsi"]["Ins"]["orig"]["path"]
    mrsi_header       = nib.load(mrsi_ref_img_path).header
    transform_list    = mridata.get_transform("inverse","mrsi")
    # transform_list    = mridata.get_transform("forward","spectroscopy") # better alignment with forward instead inverse
    parcel_image_mrsi = reg.transform(mrsi_ref_img_path,anat_outpath,transform_list,interpolator_mode="genericLabel")
    #
    prefix_name       =  prefix_name.replace("orig","mrsi")
    mrsi_outpath      = join(outdir_path,f"{prefix_name}.nii.gz")
    ftools.save_nii_file(parcel_image_mrsi.numpy(),mrsi_header,mrsi_outpath)
    debug.success("Saved parcel image in MRSI space to",mrsi_outpath)
    parc.create_tsv(labels,indices,join(outdir_path,f"{prefix_name}.tsv"))

if __name__ == "__main__":
    main()









    






