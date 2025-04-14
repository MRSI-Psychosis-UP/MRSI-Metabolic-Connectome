import os, sys, glob
from os.path import join, split, exists
import nibabel as nib
from tools.debug import Debug 
from tools.datautils import DataUtils
import re
from pathlib import Path
import numpy as np
from registration.registration import Registration
from tools.filetools import FileTools



debug  = Debug(verbose=False)
dutils = DataUtils()
reg    = Registration()
ftools = FileTools()

STRUCTURE_PATH = dutils.BIDS_STRUCTURE_PATH
subject_id_exc_list = ["CHUVA016","CHUVA028"]
METABOLITES         = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]



class DynamicData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DynamicData(**value)
            setattr(self, key, value)

class  MRIData:
    def __init__(self, subject_id,session,group):
        debug.info("dutils.BIDSDATAPATH",dutils.BIDSDATAPATH)
        self.ROOT_PATH           = join(dutils.BIDSDATAPATH,group)
        self.PARCEL_PATH         = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.CONNECTIVITY_PATH   = join(self.ROOT_PATH,"derivatives","connectivity")     
        self.DERIVATIVE_PATH     = join(self.ROOT_PATH,"derivatives")
        self.PARCEL_PATH         = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.TRANSFORM_PATH      = join(self.ROOT_PATH,"derivatives","transforms","ants")
        self.metabolites         = np.array(METABOLITES)
        self.session             = session
        self.subject_id          = subject_id
        self.prefix          = f"sub-{self.subject_id}_ses-{self.session}"
            
    def get_mri_filepath(self, modality, space, desc, met=None, option=None, acq="memprage", run="01", dwi_options=None):
        """
        Returns the path of an MRI file using BIDS keys with a standardized naming pattern.

        Expected filename patterns:
        - MRSI:  sub-<sub>_ses-<ses>_space-<space>[_met-<met>]_desc-<desc>_mrsi.nii.gz
        - T1w:   sub-<sub>_ses-<ses>_run-<run>_acq-<acq>_desc-<desc>[_T1w].nii.gz
        - DWI:   Depending on the specified dwi_options ("bval", "bvec", "mean_b0", "dwi.mif", "dwi.nii")

        Args:
            modality (str): Modality type ("mrsi", "t1w", "dwi", "func").
            space (str): Image space (e.g., "orig", "t1w", "mni").
            desc (str): Descriptor (e.g., "signal", "crlb", "brainmask", "brain").
            met (str, optional): Metabolite name (e.g., "CrPCr", "GluGln", etc.). Defaults to None.
            option (str, optional): Additional preprocessing tag (e.g., "filt_neuralnet"). Defaults to None.
            acq (str, optional): Acquisition type (default "memprage").
            run (str, optional): Run number (default "01").
            dwi_options (str, optional): Option for DWI file ("bval", "bvec", "mean_b0", "dwi.mif", "dwi.nii"). Defaults to None.

        Returns:
            str: The file path matching the pattern if found; otherwise returns a fallback path or None.
        """
        bids_root = self.ROOT_PATH
        sub, ses = self.subject_id, self.session

        # Setup based on modality
        if modality == "mrsi":
            base_dir = os.path.join(bids_root, "derivatives", f"mrsi-{space}", f"sub-{sub}", f"ses-{ses}")
            if met:
                pattern = f"sub-{sub}_ses-{ses}_space-{space}_met-{met}_desc-{desc}"
                if option:
                    pattern += f"_{option}"
            else:
                pattern = f"sub-{sub}_ses-{ses}_space-{space}_desc-{desc}"
            pattern += "_mrsi.nii.gz"
        elif modality == "t1w":
            base_dir = os.path.join(bids_root, "derivatives", "skullstrip", f"sub-{sub}", f"ses-{ses}")
            if space == "orig" and acq:
                pattern = f"sub-{sub}_ses-{ses}_run-{run}_acq-{acq}_desc-{desc}_T1w.nii.gz"
                candidate = os.path.join(base_dir, pattern)
                if not os.path.exists(candidate):
                    # Fallback if file with T1w suffix is missing.
                    pattern = f"sub-{sub}_ses-{ses}_run-{run}_acq-{acq}_desc-{desc}.nii.gz"
                    candidate = os.path.join(base_dir, pattern)
                    if not os.path.exists(candidate) and "mask" in desc:
                        pattern = f"sub-{sub}_ses-{ses}_run-{run}_acq-{acq}_desc-brain_mask.nii.gz"
            else:
                # Only 'orig' space is supported for t1w.
                debug.warning("only orig space available for t1w")
                return None
        elif modality == "dwi":
            base_dir = os.path.join(bids_root, f"sub-{sub}", f"ses-{ses}", "dwi")
            if not os.path.exists(base_dir):
                return None
            for filename in os.listdir(base_dir):
                if dwi_options == "bval" and "dwi.bval" in filename:
                    return os.path.join(base_dir, filename)
                if dwi_options == "bvec" and "dwi.bvec" in filename:
                    return os.path.join(base_dir, filename)
                if dwi_options == "mean_b0" and "mean_b0" in filename:
                    return os.path.join(base_dir, filename)
                if dwi_options == "dwi.mif" and "dwi.mif" in filename:
                    return os.path.join(base_dir, filename)
                if dwi_options == "dwi.nii" and "dwi.nii.gz" in filename:
                    return os.path.join(base_dir, filename)
            return None  # No matching DWI file found.
        else:
            debug.warning(f"Modality '{modality}' is not supported.")
            return None

        # Construct search path and find matching file(s).
        search_path = os.path.join(base_dir, pattern)
        debug.info("get_mri_filepath", search_path)
        matches = glob.glob(search_path)
        
        if len(matches) == 1:
            return matches[0]
        elif matches:
            debug.warning(f"Multiple files found for pattern {pattern}. Returning the first one.")
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

    def get_parcel_path(self,space,parc_scheme,scale,acq=None,run="01",grow=2):
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
        dirpath      = self.get_mri_parcel_dir_path("anat")
        # prefix_name  = f"{self.prefix}_run-{run}_acq-{acq}_space-{space}_atlas-{parc_scheme}_dseg.nii.gz"
        if acq is not None:
            prefix_name  = f"{self.prefix}_run-{run}_acq-{acq}_space-{space}_atlas-chimera{parc_scheme}_desc-scale{scale}grow{grow}mm_dseg.nii.gz"
        else:
            prefix_name  = f"{self.prefix}_run-{run}_space-{space}_atlas-chimera{parc_scheme}_desc-scale{scale}grow{grow}mm_dseg.nii.gz"           
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
 
    def get_connectivity_dir_path(self,modality="mrsi"):
        path = join(self.CONNECTIVITY_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            debug.warning("connectivity path does not exists")
            debug.warning(path)
            debug.success("creating connectivity path",path)
            os.makedirs(path,exist_ok=True)
            return path

    def get_connectivity_path(self,mode,parc_scheme,scale,npert=50,filtoption=""):
        dirpath     = self.get_connectivity_dir_path(mode)
        if mode=="mrsi":
            filename = f"{self.prefix}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{filtoption}_desc-connectivity_mrsi.npz"
        elif mode == "dwi":
            # filename = f"{self.prefix}_run-01_acq-memprage_space-spectroscopy_atlas-{parc_scheme}-cer-wmgeom18_dseg_connectivity.npz"
            filename = f"{self.prefix}_atlas-chimera{parc_scheme}_scale{scale}_desc-connectivity_t1w.npz"
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
  
    def get_tractography_path(self,filtered=True):
        if filtered:
            filename = "smallerTracks_200k.tck"
        else:
            filename = "tracts.tck"
        return join(self.TCK_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",filename)

    @staticmethod
    def extract_metadata(filename):
        """
        Extract metadata from a filename with (a subset of) the expected pattern:
        
        sub-SUB_ses-SES_run-RUN_acq-ACQ_space-SPACE_atlas-chimeraPARCSCHEME_desc-scaleSCALEgrowGROWmm_dseg.nii.gz

        Extracts the following fields:
        - sub (str): Subject ID.
        - ses (str): Session ID.
        - run (str): Run number.
        - acq (str, optional): Acquisition type (may be missing).
        - space (str): Image space.
        - parcscheme (str): Parcellation scheme.
        - scale (int, optional): Scale value.
        - grow (int, optional): Grow value.

        If any of these fields are not present, they are returned as None.

        Args:
            filename (str): The filename or full path to extract metadata from.

        Returns:
            dict: A dictionary with keys "sub", "ses", "run", "acq", "space", "parcscheme", "scale", "grow".
        """
        base_filename = os.path.basename(filename)
        results = {}
        
        # Define regex patterns for each field.
        patterns = {
            'sub':       r"sub-([^_]+)",
            'ses':       r"ses-([^_]+)",
            'run':       r"run-([^_]+)",
            'acq':       r"acq-([^_]+)",
            'space':     r"space-([^_]+)",
            'parcscheme': r"atlas-chimera([^_]+)",
            'scale':     r"scale(\d+)",
            'grow':      r"grow(\d+)mm",
            'npert':     r"npert(\d+)_",
            'filt':      r"filt-([^_]+)",
        }
        
        for key, pat in patterns.items():
            match = re.search(pat, base_filename)
            results[key] = match.group(1) if match else None

        # Convert numeric fields to integers if they were found.
        if results['scale'] is not None:
            results['scale'] = int(results['scale'])
        if results['grow'] is not None:
            results['grow'] = int(results['grow'])
        if results['filt'] is not None:
            results['filt'] = results['filt']
        return results  

    def find_nifti_paths(self, acq_patterns):
        """
        Searches the given BIDS directory for NIfTI (.nii) files that match the following criteria:
        - Their filename contains the subject/session pattern: "sub-<subject>_ses-<session>"
        - Their filename also contains (as a complete token) one of the acquisition patterns.
        
        Parameters:
            acq_patterns (list of str): A list of acquisition keyword patterns to match.
                (Can be provided, e.g. ["acq-memprage_desc-brain_T1w", "memprage_desc"])
        
        Returns:
            List[Path]: A list of Path objects corresponding to the files that match the criteria.
        """
        subject, session = self.subject_id, self.session
        if acq_patterns is None:
            raise Exception("No pattern specified")
        
        # Construct the basic subject/session pattern.
        prefix = f"sub-{subject}_ses-{session}"
        
        bids_dir = Path(self.ROOT_PATH)
        matched_files = []
        
        # Recursively search for .nii files.
        for nifti_file in bids_dir.rglob("*.nii*"):
            # Check if the file name contains the subject/session pattern.
            if prefix not in nifti_file.name:
                continue           
            if acq_patterns in nifti_file.name:
                matched_files.append(nifti_file)
                return nifti_file
        
        



if __name__=="__main__":
    pass