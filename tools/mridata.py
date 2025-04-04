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
    def __init__(self, subject_id,session,group=None,t1_pattern="_run-01_acq-memprage_"):
        debug.info("dutils.BIDSDATAPATH",dutils.BIDSDATAPATH)
        self.ROOT_PATH           = join(dutils.BIDSDATAPATH,group)
        self.PARCEL_PATH         = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.CONNECTIVITY_PATH   = join(self.ROOT_PATH,"derivatives","connectomes")
        self.TCK_PATH            = join(self.ROOT_PATH,"derivatives","tractography")
        self.t1_pattern          = t1_pattern
        self._mrsi_derivatives   = ["mrsi-orig","mrsi-origfilt","mrsi-t1w","mrsi-mni"]
        os.makedirs(self.PARCEL_PATH,exist_ok=True)
        os.makedirs(self.CONNECTIVITY_PATH,exist_ok=True)

        self.subject_id      = subject_id
        self.session         = session
        self.prefix          = f"sub-{subject_id}_ses-{session}"
        self.data            = json.load(open(STRUCTURE_PATH))
        self.DERIVATIVE_PATH = join(self.ROOT_PATH,"derivatives")
        self.PARCEL_PATH     = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.TRANSFORM_PATH  = join(self.ROOT_PATH,"derivatives","transforms","ants")

        self.load_mrsi_all()
        self.load_t1w(t1_pattern)
        self.load_dwi_all()
        self.load_parcels()
        self.load_connectivity("dwi")
        self.load_connectivity("mrsi")
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
            base_dir = os.path.join(bids_root, "derivatives", "mrsi-orig1", f"sub-{sub}", f"ses-{ses}")
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
            else:
                debug.warning("only orig space available for t1w")
        
        search_path = os.path.join(base_dir, pattern)
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
    


    def __get_mri_dir_path(self,modality="skullstrip"):
        if "mrsi" in modality:
            path = join(self.DERIVATIVE_PATH,modality,f"sub-{self.subject_id}",f"ses-{self.session}")
        else:
            path = join(self.DERIVATIVE_PATH,modality,f"sub-{self.subject_id}",f"ses-{self.session}")
        if os.path.exists(path):
            return path
        else:
            debug.warning("__get_mri_dir_path: path does not exists")
            debug.warning(path)
            return 
        

        
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
        
    def load_dwi_all(self):
        dirpath = self.__get_mri_dir_path("dwi")
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        for filename in filenames:
            if "dwi.bval" in filename:
                self.data["dwi"]["bval"] = join(dirpath,filename)
            elif "dwi.bvec" in filename:
                self.data["dwi"]["bvec"] = join(dirpath,filename)
            elif "dwi.nii.gz" in filename:
                self.data["dwi"]["nifti"] = join(dirpath,filename)
            elif "dwi.mif" in filename:
                self.data["dwi"]["mif"] = join(dirpath,filename)     


    def load_mrsi_all(self):
        for derivative in self._mrsi_derivatives:
            self.__load_mrsi_all(derivative)
             
    def __load_mrsi_all(self,derivative):
        dirpath = self.__get_mri_dir_path(derivative)
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        for filename in filenames:
            if ".nii" in filename:
                space = self.__extract_suffix(filename,"space")
                acq   = self.__extract_suffix(filename,"acq")
                desc  = self.__extract_suffix(filename,"desc")
                if desc in METABOLITES and acq=="conc":
                    comp = desc
                elif desc in METABOLITES and acq=="crlb":
                    comp = f"{desc}-crlb"
                elif desc == "Voxel" and acq=="fwhm":
                    comp = "fwhm"
                elif desc == "Voxel" and acq=="snr":
                    comp = "snr"
                else:
                    continue
                self.data["mrsi"][comp][space]["nifti"] = nib.load(join(dirpath,filename))
                self.data["mrsi"][comp][space]["path"]  = join(dirpath,filename)
        self.get_mrsi_mask_image()
    
    def get_mrsi_volume(self,comp,space):
        """""
        Args:
            comp : CrPCr, Ins, GluGln, NAANAAG, GPCPCh, IDEM-clrb, brainmask.
            space: orig, origfilt, t1w, mni.        
        Returns:
            NIFTI image
        """""
        path = self.get_path("mrsi",comp,space)
        try:
            _ = nib.load(path).get_fdata()
            return nib.load(path)
        except Exception as e:
            if space=="orig" or space=="origfilt":
                __nifti        = self.data["mrsi"][comp][space]["nifti"]
                return __nifti
            elif space=="t1w":
                t1w_ref        = self.data["t1w"]["brain"]["orig"]["path"]
                transform_list = self.get_transform("forward","mrsi")
                if "snr" in comp or "fwhm" in comp or "crlb" in comp:
                    _space = "orig"
                else: _space = "origfilt"
                mrsi_orig_path = self.data["mrsi"][comp][_space]["path"]
                # print("get_mrsi_volume ",mrsi_orig_path)
                mrsi_anat_np   = reg.transform(t1w_ref,mrsi_orig_path,transform_list).numpy()
                header         = nib.load(t1w_ref).header
                return ftools.numpy_to_nifti( mrsi_anat_np, header)
            elif space=="dwi":
                dwi_ref        = self.data["dwi"]["nifti"].replace("_dwi.nii.gz","_dwi_mean_b0.nii.gz")
                transform_list = self.get_transform("inverse","dwi")
                mrsi_t1w_path  = self.data["mrsi"][comp]["t1w"]["path"]
                # print("get_mrsi_volume ",mrsi_orig_path)
                mrsi_dwi_np    = reg.transform(dwi_ref,mrsi_t1w_path,transform_list).numpy()
                header         = nib.load(dwi_ref).header
                return ftools.numpy_to_nifti( mrsi_dwi_np, header)
            elif space=="mni":
                mrsi_anat_nii  = self.get_mrsi_volume(comp,"t1w")
                mni_ref        = datasets.load_mni152_template()
                transform_list = self.get_transform("forward","anat")
                mrsi_mni_np    = reg.transform(mni_ref,mrsi_anat_nii,transform_list).numpy()
                header         = mni_ref.header
                return ftools.numpy_to_nifti( mrsi_mni_np, header)


    def get_mrsi_mask_image(self):
        for derivative in self._mrsi_derivatives:
            self.__get_mrsi_mask_image(derivative)
            
    def __get_mrsi_mask_image(self,derivative):
        dirpath   = self.__get_mri_dir_path(derivative)
        if dirpath is None :return
        filenames = os.listdir(dirpath)
        mrsi_mask = None
        if len(filenames)==0:
            return
        for filename in filenames:
            path = join(dirpath,filename)
            if "WaterSignal_mrsi.nii.gz" in filename:
                water_signal  = nib.load(path).get_fdata().squeeze()
                header        = nib.load(path).header
                mrsi_mask     = np.zeros(water_signal.shape)
                mrsi_mask[water_signal>0]  = 1
                mrsi_mask[water_signal<=0] = 0
                #
                space       = self.__extract_suffix(filename,"space")
                outfilename = f"{self.prefix}_space-{space}_acq-conc_desc-brainmask_mrsi.nii.gz"
                outpath     = join(dirpath,outfilename)
                #
                ftools.save_nii_file(mrsi_mask,header,outpath) 
                nifti_img = ftools.numpy_to_nifti(mrsi_mask,header)
                self.data["mrsi"]["mask"][space]["nifti"]     = nifti_img
                self.data["mrsi"]["mask"][space]["path"]      = outpath
                if space=="orig":
                    outpath = outpath.replace("orig","origfilt")
                    os.makedirs(split(outpath)[0],exist_ok=True)
                    ftools.save_nii_file(mrsi_mask,header,outpath) 
                    self.data["mrsi"]["mask"]["origfilt"]["nifti"] = nifti_img
                    self.data["mrsi"]["mask"]["origfilt"]["path"]  = outpath



    def get_path(self,modality,comp,space):
        """""
        Args:
            modality: mrsi
            comp : CrPCr, Ins, GluGln, NAANAAG, GPCPCh, IDEM-clrb, brainmask.
            space: orig, origfilt, t1w, mni.        
        Returns:
            NIFTI path
        """""

        filename = f"sub-{self.subject_id}_ses-{self.session}"
        if modality=="mrsi":
            derivative = f"{modality}-{space}"
            dir_path = self.__get_mri_dir_path(derivative)
            if "crlb" in comp:
                acq = "crlb"
                desc = comp.replace("-crlb","")
            elif comp in METABOLITES or comp == "brainmask" or comp == "mask":
                acq = "conc"
                desc = comp
            elif comp == "snr" or comp == "fwhm":
                acq = comp
                desc = "Voxel"
            else:
                debug.error("MRIData:get_path unrecognized component")
                return
            filename += f"_space-{space}_acq-{acq}"
            filename += f"_desc-{desc}_mrsi.nii.gz"

        else:
            debug.error("MRIData:get_path unrecognized modality")
            return
        return join(dir_path,filename)
    
    def load_t1w(self,pattern=""):
        dirpath = self.__get_mri_dir_path("skullstrip")
        debug.info("load_t1w",dirpath)
        if dirpath is None or len(os.listdir(dirpath))==0:
            return 
        filenames = os.listdir(dirpath)
        for filename in filenames:
            debug.info("load_t1w:filename",filename)
            if pattern not in filename:
                continue
            path = join(dirpath,filename)
            if "brain" in filename and "mask" not in filename:
                debug.success("Found T1w_brain",filename)
                # self.data["t1w"]["brain"]["orig"]["nifti"] = nib.load(path)
                self.data["t1w"]["brain"]["orig"]["path"]  = path
            elif "mask" in filename:
                debug.success("Found T1w_brainmask",filename)
                # self.data["t1w"]["mask"]["orig"]["nifti"] = nib.load(path)
                self.data["t1w"]["mask"]["orig"]["path"] = path
        if self.data["t1w"]["mask"]["orig"]["path"]==0:
            t1wbrain_path = self.data["t1w"]["brain"]["orig"]["path"]
            t1wmask_path  = t1wbrain_path.replace("T1w_brain","T1w_brainmask")
            t1w_brain_nii = nib.load(t1wbrain_path)
            t1w_brain_np  = t1w_brain_nii.get_fdata()
            mask          =  np.zeros(t1w_brain_np.shape) 
            mask[t1w_brain_np>0] = 1
            ftools.save_nii_file(mask,t1w_brain_nii.header,t1wmask_path)
            self.data["t1w"]["mask"]["orig"]["path"] = t1wmask_path

    def get_t1w(self,pattern="_run-01_acq-memprage_"):
        dirpath = self.__get_mri_dir_path("skullstrip")
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        for filename in filenames:
            if pattern in filename and "nii.gz" in filename:
                path = join(dirpath,filename)
                return nib.load(path), path
        debug.error(f"get_t1w {pattern} not found")
        return 
    

    

    def get_connectivity_path(self,mode,atlas):
        dirpath     = self.get_connectivity_dir_path(mode)
        prefix_name = f"{self.prefix}_run-01_acq-memprage_atlas-{atlas}_connectivity.npz"
        if mode == "dwi":
            scheme,scale  = atlas[0:9],atlas[-1]
            prefix_name = f"{self.prefix}_run-01_acq-memprage_space-mrsi_atlas-{atlas}-cer_dseg_connectivity.npz"
            # prefix_name = f"{self.prefix}_atlas-chimera{scheme}_desc-scale{scale}grow2mm_dseg_connectivity.npz"
        elif mode=="mrsi":
            prefix_name = f"{self.prefix}_run-01_acq-memprage_atlas-{atlas}_connectivity.npz"
        return join(dirpath,prefix_name)
 


    def get_tractography_path(self,filtered=True):
        if filtered:
            filename = "smallerTracks_200k.tck"
        else:
            filename = "tracts.tck"
        return join(self.TCK_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",filename)
    


    def load_parcels(self):
        dirpath = self.get_mri_parcel_dir_path("anat")
        # debug.info("load_parcels:dirpath",dirpath)
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        debug.info("load_parcels:found n",len(filenames),"Filenames")
        if len(filenames)==0:
            return
        filename = filenames[0]
        for filename in filenames:
            path = join(dirpath,filename)
            if ".nii.gz" in filename and "dseg" in filename and "wm_mask" not in filename:
                space = self.__extract_suffix(filename,"space")
                # debug.info("load_parcels:filename",filename)
                # if space == "orig":
                for atlas in ATLAS_PARC_LIST:
                    if atlas in filename and atlas!="chimera":
                        debug.info("load_parcels:Found",atlas,space,filename)
                        try:
                            self.data["parcels"][atlas][space]["path"]      = path
                            self.data["parcels"][atlas][space]["labelpath"] = path.replace("nii.gz","tsv")
                        except:pass
                    elif atlas in filename and atlas=="chimera":
                        scale,scheme  = self.__extract_scale_number(filename)
                        try:
                            self.data["parcels"][f"{scheme}-{scale}"][space]["path"]      = path
                            self.data["parcels"][f"{scheme}-{scale}"][space]["labelpath"] = path.replace("nii.gz","tsv")
                        except:pass

    def load_connectivity(self,mode="dwi"):
        dirpath = self.get_connectivity_dir_path(mode)
        # debug.info("load_parcels:dirpath",dirpath)
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        # filename = filenames[0]
        for filename in filenames:
            # debug.info("load_parcels:filename",filename)
            path = join(dirpath,filename)
            if "_simmatrix.npz" in path:
                for atlas in ATLAS_LIST:
                    if atlas in filename and atlas!="chimera":
                        self.data["connectivity"][mode][atlas]["path"] = path
                    elif atlas in filename and atlas=="chimera":
                        parc_scheme = self.__extract_parcellation_substring(filename)
                        self.data["connectivity"][mode][parc_scheme]["path"] = path 

    
    def __extract_suffix(self,filename,suffix):
        # Use a regular expression to search for the pattern matching 'space-{SPACE}'
        match = re.search(rf"_{suffix}-([^_]+)", filename)
        if match:
            return match.group(1)  # Returns the captured group, which corresponds to {SPACE}
        else:
            return None  # Return None if the pattern is not found

    def __extract_scale_number(self,filename):
        """
        Extracts the number following 'scale' in the chimera filename.

        Args:
        filename (str): The filename from which to extract the scale number.

        Returns:
        int: The number following 'scale' or None if no such number is found.
        """
        # Define the regular expression to find 'scale' followed by any number
        match_scale = re.search(r"scale(\d+)", filename)
        match_scheme = re.search(r"atlas-chimera(\w+)", filename)
        # debug.info("__extract_scale_number:match_scheme",match_scheme)
        # Extract the scale number
        scale_number = int(match_scale.group(1)) if match_scale else None

        # Extract the scheme
        scheme = match_scheme.group(1) if match_scheme else None
        scheme = scheme.replace("_desc","")
        return scale_number, scheme

    def __extract_parcellation_substring(self,input_string):
        # Define regex patterns for the two types of substrings
        pattern1 = r'geometric_cubeK23mm'
        pattern2 = r'geometric_cubeK18mm'
        pattern3 = r'chimeraLFIIHIFIF\d+'
        pattern4 = r'chimeraLFMIHIFIF\d+'
        pattern5 = r'wm_cubeK18mm\d+'
        pattern6 = r'wm_cubeK15mm\d+'
        
        # Search for the patterns in the input string
        match1 = re.search(pattern1, input_string)
        match2 = re.search(pattern2, input_string)
        match3 = re.search(pattern3, input_string)
        match4 = re.search(pattern4, input_string)
        match5 = re.search(pattern5, input_string)
        match6 = re.search(pattern6, input_string)
       
        # Return the matched substring
        if match1:
            return match1.group()
        if match2:
            return match2.group()
        if match5:
            return match5.group()
        if match6:
            return match6.group()
        elif "LFIIHIFIF" in input_string or "chimeraLFMIHIFIF" in input_string:
            scale,scheme  = self.__extract_scale_number(input_string)
            return f"{scheme}-{scale}"
        else:
            return None




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
