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
    parser.add_argument('--parc', type=str, default="LFMIHIFIS", 
                        help='Base Chimera parcellation string (must be one of: LFMIHIFIF)')
    parser.add_argument('--scale',type=int,default=3,help="Cortical parcellation scale (default: 3)")

    parser.add_argument('--overwrite',type=int,default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--group', type=str,default="Dummy-Project") 
    parser.add_argument('--subject_id', type=str, help='subject id', default="S001")
    parser.add_argument('--session', type=str, help='recording session',choices=['V1', 'V2', 'V3','V4','V5'], default="V1")

    args        = parser.parse_args()
    parc_scheme = args.parc
    overwrite   = args.overwrite
    scale       = args.scale
    GROUP       = args.group
    ftools      = FileTools()



    ############ Set up Recording ID ##################
    subject_id, session = args.subject_id, args.session
    mridata = MRIData(subject_id,session,group=GROUP)
    
    parcel_mrsi_path  = mridata.get_parcel_path(parc_scheme=parc_scheme,scale=scale,space="mrsi")
    # Map T1 parcellation to MRSI space
    if exists(parcel_mrsi_path) and not overwrite:
        debug.success("Parcellation image already mapped to MRSI space")
        return
    elif not exists(parcel_mrsi_path) or overwrite:
        # mrsi_ref_img_path = mridata.data["mrsi"]["Ins"]["orig"]["path"]
        parcel_anat_path   = mridata.get_parcel_path(parc_scheme=parc_scheme,scale=scale,space="orig")
        mrsi_ref_img_nifti = mridata.get_mri_nifti(modality="mrsi",space="orig",desc="signal",
                                                        met="CrPCr",option="filtbiharmonic")
        mrsi_header       = mrsi_ref_img_nifti.header
        transform_list    = mridata.get_transform("inverse","mrsi")
        parcel_image_mrsi = reg.transform(mrsi_ref_img_nifti,parcel_anat_path,transform_list,
                                            interpolator_mode="genericLabel")
        #
        
        ftools.save_nii_file(parcel_image_mrsi.numpy(),mrsi_header,parcel_mrsi_path)
        debug.success("Saved parcel image in MRSI space to",parcel_mrsi_path)
        indices, labels, _  = parc.read_tsv_file(parcel_anat_path.replace(".nii.gz",".tsv"))
        parc.create_tsv(labels,indices,parcel_mrsi_path.replace(".nii.gz",".tsv"))

if __name__ == "__main__":
    main()









    






