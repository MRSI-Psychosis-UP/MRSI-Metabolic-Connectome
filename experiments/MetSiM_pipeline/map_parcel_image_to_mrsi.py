import os, sys, json
import numpy as np
from mrsitoolbox.registration.registration import Registration
from mrsitoolbox.tools.datautils import DataUtils
from os.path import split, join, exists
from mrsitoolbox.tools.filetools import FileTools
from mrsitoolbox.tools.debug import Debug
from os.path import join, split
from mrsitoolbox.tools.mridata import MRIData
from mrsitoolbox.connectomics.parcellate import Parcellate
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
                        help='Base Chimera parcellation string (must be one of: LFMIHIFIS,LFMIHIFIF)')
    parser.add_argument('--scale',type=int,default=3,help="Cortical parcellation scale (default: 3)")
    parser.add_argument('--grow',type=int,default=2,help="Gyral WM grow into GM in mm (default: 2)")
    parser.add_argument('--overwrite',type=int,default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
    parser.add_argument('--group', type=str,default="Dummy-Project") 
    parser.add_argument('--subject_id', type=str, help='subject id', default="S001")
    parser.add_argument('--session', type=str, help='recording session', default="V1")
    parser.add_argument('--t1' , type=str, default="desc-brain_T1w",help="Anatomical T1w file pattern or full path")

    args        = parser.parse_args()
    parc_scheme = args.parc
    overwrite   = args.overwrite
    scale       = args.scale
    GROUP       = args.group
    ftools      = FileTools()
    t1pattern   = args.t1
    grow_mm     = args.grow





    ############ Set up Recording ID ##################
    subject_id, session = args.subject_id, args.session
    mridata = MRIData(subject_id,session,group=GROUP)

    # Check T1W path if specfied
    if t1pattern is not None:
        if not exists(t1pattern):
            t1_img_path = mridata.find_nifti_paths(t1pattern)
        else:
            t1_img_path = t1pattern
        if t1_img_path is not None:
            msg = f"Found t1 image: {t1_img_path}"
            debug.info(msg)
        else:
            msg = f"Could not find a t1 image matching the pattern [{t1pattern}]. Skipping..."
            debug.error(GROUP, subject_id, session, msg)
            return
    else:
        t1_img_path       = mridata.get_mri_filepath("T1w","orig","brain")


    metadata_dict = mridata.extract_metadata(t1_img_path)
    t1_res     = np.array(nib.load(t1_img_path).header.get_zooms()[:3]).mean()
    mni_t1_res = datasets.load_mni152_template(t1_res)
    print("metadata_dict",metadata_dict)
    parcel_mrsi_path  = mridata.get_parcel_path(parc_scheme=parc_scheme,scale=scale,space="mrsi",
                                                acq=metadata_dict["acq"],run=metadata_dict["run"],
                                                grow=grow_mm)
    debug.info("parcel_mrsi_path",exists(parcel_mrsi_path),parcel_mrsi_path)
    # Map T1 parcellation to MRSI space
    if exists(parcel_mrsi_path) and not overwrite:
        debug.success("Parcellation image already mapped to MRSI space")
        return
    elif not exists(parcel_mrsi_path) or overwrite:
        parcel_anat_path   = mridata.get_parcel_path(parc_scheme=parc_scheme,scale=scale,space="orig",
                                                     acq=metadata_dict["acq"],run=metadata_dict["run"])
        debug.info("parcel_anat_path",exists(parcel_anat_path))
        if "cubic" in parc_scheme:
            parcel_mni_path   = mridata.get_parcel_path(parc_scheme="cubic",scale=scale,space="mni",
                                                        acq=metadata_dict["acq"],run=metadata_dict["run"])
            parcel_image_np, labels, indices, header = parc.create_parcel_image(atlas_string=f"{parc_scheme}{scale}")
            ftools.save_nii_file(parcel_image_np,header=header,outpath=parcel_mni_path)
            # transform to T1w space
            transform_list    = mridata.get_transform("inverse","anat")
            
            parcel_anat       = reg.transform(t1_img_path,parcel_mni_path,transform_list,
                                                interpolator_mode="genericLabel")
            t1_header         = nib.load(t1_img_path).header
            ftools.save_nii_file(parcel_anat.numpy(),header=t1_header,outpath=parcel_anat_path)
            parc.create_tsv(labels,indices,parcel_anat_path.replace(".nii.gz",".tsv"))
        # transform to MRSI space
        debug.proc("Transform parcel image to MRSI space")
        mrsi_ref_img_nifti = mridata.get_mri_nifti(modality="mrsi",space="orig",desc="signal",
                                                        met="CrPCr",option="filtbiharmonic")
        mrsi_header       = mrsi_ref_img_nifti.header
        transform_list    = mridata.get_transform("inverse","mrsi")
        parcel_image_mrsi = reg.transform(mrsi_ref_img_nifti,parcel_anat_path,transform_list,
                                            interpolator_mode="genericLabel")
        #
        debug.info("Save parcel_image_mrsi")
        ftools.save_nii_file(parcel_image_mrsi.numpy(),header=mrsi_header,outpath=parcel_mrsi_path)
        debug.success("Saved parcel image in MRSI space to",parcel_mrsi_path)
        indices, labels, _  = parc.read_tsv_file(parcel_anat_path.replace(".nii.gz",".tsv"))
        parc.create_tsv(labels,indices,parcel_mrsi_path.replace(".nii.gz",".tsv"))
        # transform to MNI space
        debug.proc("Transform parcel image to MNI152 space")
        transform_list    = mridata.get_transform("forward","anat")
        # transform_list   += mridata.get_transform("forward","mrsi")
        parcel_image_mrsi = reg.transform(mni_t1_res,parcel_anat_path,transform_list,
                                            interpolator_mode="genericLabel")
        
        parcel_mni_path = parcel_mrsi_path.replace("space-mrsi","space-mni152")
        ftools.save_nii_file(parcel_image_mrsi.numpy(),header=mni_t1_res.header,outpath=parcel_mni_path)
        parc.create_tsv(labels,indices,parcel_mni_path.replace(".nii.gz",".tsv"))
        debug.info("Saved results to ",parcel_mni_path)

if __name__ == "__main__":
    main()









    






