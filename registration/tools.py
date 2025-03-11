
from tools.debug import Debug
import os , glob
import shutil
from os.path import split, join
import numpy as np

debug  = Debug()


class RegTools:
    def __init__(self) -> None:
        pass

 
    def save_all_transforms(self,ants_transform_list,dir_prefix_path):
        os.makedirs(split(dir_prefix_path)[0],exist_ok=True)
        forward_transform_list = ants_transform_list["fwdtransforms"]
        for transform_path in forward_transform_list:
            filename = split(transform_path)[1]
            if "Warp" in filename:
                outpath=f"{dir_prefix_path}.syn.nii.gz"
            elif "Affine" in filename:
                outpath=f"{dir_prefix_path}.affine.mat"
            shutil.copy(transform_path, outpath)
        inverse_transform_list = ants_transform_list["invtransforms"]
        for transform_path in inverse_transform_list:
            filename = split(transform_path)[1]
            if "Warp" in filename:
                outpath=f"{dir_prefix_path}.syn_inv.nii.gz"
            elif "Affine" in filename:
                outpath=f"{dir_prefix_path}.affine_inv.mat"
            shutil.copy(transform_path, outpath)
        debug.success("Saved all transforms to ",split(dir_prefix_path)[1])



    def get_transform(self,direction,space,metabolite_ref="CrPCr"):
        self.ROOT_PATH            = join(self.TRANSFORM_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",space)

        if space=="spectroscopy":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-mrsi_{metabolite_ref}_to_t1w"
        elif  space=="anat":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-t1w_to_mni"

        transform_list = list()
        if direction=="forward":  
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.syn.nii.gz"))
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.affine.mat"))
        elif  direction=="inverse":
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.affine_inv.mat"))
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.syn_inv.nii.gz"))
        return transform_list


