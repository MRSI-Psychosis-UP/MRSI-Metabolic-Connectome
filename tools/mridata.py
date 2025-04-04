import os, sys, glob
from os.path import join, split, exists
import nibabel as nib
from tools.debug import Debug 
from tools.datautils import DataUtils
import re
import numpy as np
import json
from registration.registration import Registration
from tools.filetools import FileTools
from nilearn import datasets
import glob



dutils = DataUtils()
reg    = Registration()
ftools = FileTools()
debug  = Debug(verbose=False)

STRUCTURE_PATH = dutils.BIDS_STRUCTURE_PATH
subject_id_exc_list = ["CHUVA016","CHUVA028"]
METABOLITES         = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
ATLAS_PARC_LIST          = ["aal","destrieux","jhu_icbm_wm","wm_cubeK15mm","wm_cubeK18mm",
                       "geometric_cubeK18mm","geometric_cubeK23mm","cerebellum","chimera"]
ATLAS_LIST          = ["aal","destrieux","jhu_icbm_wm","wm_cubeK15mm","wm_cubeK18mm",
                       "geometric_cubeK18mm","geometric_cubeK23mm","cerebellum","chimera"]


class DynamicData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DynamicData(**value)
            setattr(self, key, value)

class  MRIData:
    def __init__(self, subject_id,session,group=None):
        debug.info("dutils.BIDSDATAPATH",dutils.BIDSDATAPATH)
        self.ROOT_PATH           = join(dutils.BIDSDATAPATH,group)
        self.PARCEL_PATH         = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.CONNECTIVITY_PATH   = join(self.ROOT_PATH,"derivatives","connectivity")
        os.makedirs(self.PARCEL_PATH,exist_ok=True)
        os.makedirs(self.CONNECTIVITY_PATH,exist_ok=True)

        self.subject_id      = subject_id
        self.session         = session
        self.prefix          = f"sub-{subject_id}_ses-{session}"
        self.data            = json.load(open(STRUCTURE_PATH))
        self.DERIVATIVE_PATH = join(self.ROOT_PATH,"derivatives")
        self.PARCEL_PATH     = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.TRANSFORM_PATH  = join(self.ROOT_PATH,"derivatives","transforms","ants")
        self.metabolites = np.array(METABOLITES)
 


    def get_mri_filepath(self,modality, space, desc, met=None,option=None,acq="memprage",run="01"):
        """
        Returns the path of an MRSI file based on BIDS keys.
        
        The expected naming pattern is:
        sub-<sub>_ses-<ses>_space-<space>[_met-<met>]_desc-<desc>_mrsi.nii.gz
        
        Args:
            modality (str): Imaging modality (e.g, "t1w","mrsi","dwi","func")
            space (str): Image space (e.g., "orig", "t1w", "mni").
            desc (str): Descriptor of the file (e.g., "signal", "crlb", "brainmask","brain").
            met (str, optional): Metabolite name (e.g., "CrPCr","GluGln","GPCPCh","NAANAAG","Ins"). Defaults to None.
            option (str, optional): Preprocessing string (e.g., "filt_neuralnet", "filt_biharmonic"). Defaults to None.
        Returns:
            str : The matching file path if found, else None.
        """

        bids_root = self.ROOT_PATH
        sub,ses = self.subject_id,self.session
        if modality=="mrsi":
            base_dir = os.path.join(bids_root, "derivatives", "mrsi-orig", f"sub-{sub}", f"ses-{ses}")
            # Build the filename pattern. If met is provided, include the metabolite key.
            if met and option:
                pattern = f"sub-{sub}_ses-{ses}_space-{space}_met-{met}_desc-{desc}_{option}_mrsi.nii.gz"
            elif met and option is None:
                pattern = f"sub-{sub}_ses-{ses}_space-{space}_met-{met}_desc-{desc}_mrsi.nii.gz"
            else:
                pattern = f"sub-{sub}_ses-{ses}_space-{space}_desc-{desc}_mrsi.nii.gz"
        if modality=="t1w":
            base_dir = os.path.join(bids_root, "derivatives", "skullstrip", f"sub-{sub}", f"ses-{ses}")
            # Build the filename pattern. If met is provided, include the metabolite key.
            
            if space=="orig" and acq is not None:
                pattern = f"sub-{sub}_ses-{ses}_run-{run}_acq-{acq}_desc-{desc}_T1w.nii.gz"
                if not exists(join(base_dir, pattern)):
                    pattern = f"sub-{sub}_ses-{ses}_run-{run}_acq-{acq}_desc-{desc}.nii.gz"
                    if not exists(join(base_dir, pattern)) and "mask" in desc:
                        pattern = f"sub-{sub}_ses-{ses}_run-{run}_acq-{acq}_desc-brain_mask.nii.gz"



                print("get_mri_filepath",join(base_dir, pattern))
            else:
                debug.warning("only orig space available for t1w")
        
        search_path = join(base_dir, pattern)
        matches = glob.glob(search_path)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            debug.warning(f"Warning: Multiple files found for pattern {pattern}. Returning the first one.")
            return matches[0]
        else:
            debug.warning(f"No file found for pattern: {pattern}")
            return search_path

    def get_mri_nifti(self,modality, space, desc, met=None,option=None,acq="memprage",run="01"):
        """
        Returns the nibabel Nifti1Image for an MRI file based on BIDS keys.
        
        See get_mri_filepath for the expected naming patterns.
        
        Args:
            modality (str): Imaging modality (e.g., "t1w", "mrsi", "dwi", "func").
            space (str): Image space (e.g., "orig", "t1w", "mni").
            desc (str): Descriptor of the file (e.g., "signal", "crlb", "brainmask", "brain").
            met (str, optional): Metabolite name. Defaults to None.
            option (str, optional): Preprocessing string. Defaults to None.
            acq (str, optional): Acquisition parameter. Defaults to "memprage".
            run (str, optional): Run identifier. Defaults to "01".
            
        Returns:
            nibabel.Nifti1Image: The loaded image if found.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = self.get_mri_filepath(modality, space, desc, met, option, acq, run)
        if exists(path):
            return nib.load(path)
        else:
            raise FileNotFoundError(f"{split(path)[0]} does not exist")

    def get_parcel_path(self,space,parc_scheme,scale,acq="memprage",run="01",grow=2):
        """""
        Returns the path to the Chimera parcellation file.
        Args:
            parc_scheme : chimera parcellation scheme: LFMIHIFIF, LFMIHIFIS,
            scale       : cortical parcellation scale
            space (str): Image space (e.g., "orig", "t1w", "mni").
            acq (str, optional): Acquisition parameter. Defaults to "memprage".
            run (str, optional): Run identifier. Defaults to "01".
            grow (int): GM growth into WM
        Returns:
            str : The matching file path.
        """""
        if space=="mni":
            return join(dutils.DEVDATAPATH,"atlas",f"chimera-{parc_scheme}-{scale}",f"chimera-{parc_scheme}-{scale}.nii.gz")
        else:
            dirpath      = self.get_mri_parcel_dir_path("anat")
            # prefix_name  = f"{self.prefix}_run-{run}_acq-{acq}_space-{space}_atlas-{parc_scheme}_dseg.nii.gz"
            prefix_name  = f"{self.prefix}_run-{run}_acq-{acq}_space-{space}_atlas-chimera{parc_scheme}_desc-scale{scale}grow{grow}mm_dseg.nii.gz"
        return join(dirpath,prefix_name)     


    def get_mri_parcel_dir_path(self,modality="anat"):
        path = join(self.PARCEL_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            debug.warning("path does not exists")
            debug.warning(path)
            return 

    def get_parcel(self,space,parc_scheme,scale,acq="memprage",run="01",grow=2):
        path = self.get_parcel_path(space,parc_scheme,scale,acq,run,grow)
        return nib.load(path),path        

        
    def get_connectivity_dir_path(self,modality="dwi"):
        path = join(self.CONNECTIVITY_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            debug.warning("connectivity path does not exists")
            debug.warning(path)
            debug.success("creating connectivity path",path)
            os.makedirs(path,exist_ok=True)
            return path


        

    def get_connectivity_path(self,mode,parc_scheme,scale,npert=50):
        dirpath     = self.get_connectivity_dir_path(mode)
        if mode=="mrsi":
            filename = f"{self.prefix}_atlas-chimera{parc_scheme}_scale{scale}_desc-connectivity_npert{npert}_mrsi.npz"

        return join(dirpath,filename)
 

    def get_transform(self,direction,space):
        transform_dir_path            = join(self.TRANSFORM_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",space)

        if space=="mrsi":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-mrsi_to_t1w"
        elif  space=="anat":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-t1w_to_mni"
        elif  space=="dwi":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-dwi_to_t1w"

        transform_list = list()
        if direction=="forward":  
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.syn.nii.gz"))
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.affine.mat"))
        elif  direction=="inverse":
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.affine_inv.mat"))
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.syn_inv.nii.gz"))
        return transform_list




if __name__=="__main__":
    mrsiData = MRIData(subject_id="S001",session="V1",group="Dummy-Project")
