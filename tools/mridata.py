import os, sys, glob
from os.path import join, split, exists, isdir
import nibabel as nib
import re
from pathlib import Path
import numpy as np
from tools.debug import Debug 
from tools.datautils import DataUtils
from registration.registration import Registration
from tools.filetools import FileTools



debug  = Debug(verbose=False)
dutils = DataUtils()
reg    = Registration()
ftools = FileTools()

STRUCTURE_PATH = dutils.BIDS_STRUCTURE_PATH
# METABOLITES_3T         = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
# METABOLITES_7T         = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]


class DynamicData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DynamicData(**value)
            setattr(self, key, value)

class  MRIData:
    def __init__(self, subject_id="",session="",group="Dummy-Project",metabolites=["CrPCr", "GluGln", "GPCPCh", "NAANAAG", "Ins"]):
        debug.info("dutils.BIDSDATAPATH",dutils.BIDSDATAPATH)
        self.ROOT_PATH           = join(dutils.BIDSDATAPATH,group)
        self.PARCEL_PATH         = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.CONNECTIVITY_PATH   = join(self.ROOT_PATH,"derivatives","connectivity")     
        self.DERIVATIVE_PATH     = join(self.ROOT_PATH,"derivatives")
        self.TRANSFORM_PATH      = join(self.ROOT_PATH,"derivatives","transforms","ants")
        self.MSIDIRPATH          = join(self.DERIVATIVE_PATH,"group","msi","mrsi")     
        self.metabolites         = metabolites
        self.session             = session
        self.subject_id          = subject_id
        self.prefix              = f"sub-{self.subject_id}_ses-{self.session}"

            




    def get_mri_filepath(self, modality, space, desc, met=None, option=None, 
                         acq="memprage", run="01", dwi_options=None, res=None):
        """
        Returns the path of an MRI file using BIDS keys with a standardized naming pattern.

        Expected filename patterns:
        - T1w:   sub-<sub>_ses-<ses>_run-<run>_acq-<acq>_desc-<desc>[_T1w].nii.gz
        - DWI:   Depending on the specified dwi_options ("bval", "bvec", "mean_b0", "dwi.mif", "dwi.nii")

        Args:
            modality (str): Modality type ("mrsi", "t1w", "dwi", "func").
            space (str): Image space (e.g., "orig", "T1w", "mni","mni152long").
            res (int): Image isotropic resolution, defaults to original resolution if set to None.
            desc (str): Descriptor (e.g., "signal", "crlb","fwhm","snr", "brainmask", "brain").
            met (str, optional): Metabolite name (e.g., "CrPCr", "GluGln", etc.). Defaults to None.
            option (str, optional): Additional preprocessing tag (e.g., "filt_neuralnet"). Defaults to None.
            acq (str, optional): Acquisition type (default "memprage").
            run (str, optional): Run number (default "01").
            dwi_options (str, optional): Option for DWI file ("bval", "bvec", "mean_b0", "dwi.mif", "dwi.nii"). Defaults to None.

        Returns:
            str: The file path matching the pattern if found; otherwise returns a fallback path or None.
        """
        if not modality=="mrsi":
            imagepath = self.__get_mri_filepath( modality, space, 
                                                desc, met, option, 
                                                acq, run, dwi_options)
        else:
            imagepath = self.__get_mri_filepath(modality, space, 
                                                desc, met, option, 
                                                acq, run, dwi_options)
            if not exists(imagepath):
                path_list = list()
                if any(met == _m for _m in ["Glx","GluGln"]):
                    metlist =  ["Glu","Gln"]
                elif any(met == _m for _m in ["tNAA","NAANAAG"]):
                    metlist = ["NAA","NAAG"]
                elif any(met == _m for _m in ["tNAA","NAANAAG"]):
                    metlist  =  ["NAA","NAAG"]
                elif any(met == _m for _m in ["tCr","CrPCr"]):
                    metlist = ["CrPCr"]
                elif any(met == _m for _m in ["Cho","GPCPCh"]):    
                    metlist =  ["GPCPCh"]
                for metind in metlist:
                    _path = self.__get_mri_filepath(modality, space, 
                                                    desc, metind, option, 
                                                    acq, run, dwi_options)
                    path_list.append(_path)
                return path_list
            else:
                return imagepath
        

    def __get_mri_filepath(self, modality, space, desc, met=None, option=None, 
                         acq="memprage", run="01", dwi_options=None, res=None):
        
        bids_root = self.ROOT_PATH
        sub, ses = self.subject_id, self.session
        res_str  = "" if res is None else f"_res-{res}mm"
        # Setup based on modality
        if modality == "mrsi":
            _dirspace = (
                "orig" if space == "mrsi"
                else "T1w" if space == "anat"
                else space
            )
            base_dir = os.path.join(bids_root, "derivatives", f"mrsi-{_dirspace}", f"sub-{sub}", f"ses-{ses}")
            _space="orig" if space=="mrsi" else space
            debug.info("get_mri_filepath: looking inside",space,_dirspace)
            if met:
                pattern = f"sub-{sub}_ses-{ses}_space-{_space}{res_str}_met-{met}_desc-{desc}"
                if option:
                    pattern += f"_{option}"
            else:
                pattern = f"sub-{sub}_ses-{ses}_space-{_space}{res_str}_desc-{desc}"
            pattern += "_mrsi.nii.gz"
        elif modality.lower() == "t1w":
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



    def get_mri_nifti(self,modality, space, desc, met=None,option=None,acq="memprage",run="01",res=None):
        """
        Returns the nibabel Nifti1Image for an MRI file based on BIDS keys.
        
        See get_mri_filepath for the expected naming patterns.
        
        Args:
            modality (str): Imaging modality (e.g., "t1w", "mrsi", "dwi", "func").
            space (str): Image space (e.g., "orig", "t1w", "mni").
            desc (str): Descriptor (e.g., "signal", "crlb","fwhm","snr", "brainmask", "brain").
            met (str, optional): Metabolite name. Defaults to None.
            option (str, optional): Preprocessing string. Defaults to None.
            acq (str, optional): Acquisition parameter. Defaults to "memprage".
            run (str, optional): Run identifier. Defaults to "01".
            res (int): Image isotropic resolution, defaults to original resolution if set to None.
            
        Returns:
            nibabel.Nifti1Image: The loaded image if found. If get_mri_filepath
            returns a list (composite metabolites), their data are summed.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = self.get_mri_filepath(modality, space, desc, met, option, acq, run,res)
        if path is None:
            raise FileNotFoundError(f"{path} does not exist")

        if isinstance(path, list):
            if not all([exists(_p) for _p in path]):
                raise FileNotFoundError("One or mor NIFTI images not found for requested metabolite.")
            else:
                images = []
                for p in path:
                    images.append(nib.load(p))
                base_img = images[0]
                data_sum = np.zeros_like(base_img.get_fdata())
                for img in images:
                    if img.shape != base_img.shape:
                        raise ValueError(f"Composite metabolites require matching shapes, got {img.shape} vs {base_img.shape}")
                    data_sum += img.get_fdata()
                return nib.Nifti1Image(data_sum, base_img.affine, base_img.header)

        if exists(path):
            return nib.load(path)
        return None


    def get_pv_filepath(self, desc, acq=None, run=None):
        """
        Returns the “best” matching partial-volume NIfTI in:
            {DERIVATIVE_PATH}/cat12/sub-{subject_id}/ses-{session}/

        It will look at all .nii/.nii.gz files there, score them by how many of
        the requested tags (desc-, acq-, run-) they contain, and return the
        highest-scoring one (breaking ties alphabetically).

        Args:
            desc (str): Descriptor, e.g. "p1", "p2", "p3"  → GM/WM/CSF maps
            acq  (str, optional): e.g. "memprage"
            run  (str, optional): e.g. "01"
        Returns:
            str or None: Full path to the best match, or None if none found.
        """
        dir_path = join(
            self.DERIVATIVE_PATH,
            "cat12",
            f"sub-{self.subject_id}",
            f"ses-{self.session}"
        )

        if not isdir(dir_path):
            # no such session folder
            return None

        # Gather all NIfTI files in that directory
        candidates = [
            f for f in os.listdir(dir_path)
            if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")
        ]
        if not candidates:
            return None

        # Build up to three regex patterns for desc, acq, run
        patterns = []
        patterns.append(("desc", re.compile(rf"desc-{re.escape(desc)}", re.IGNORECASE)))
        if acq:
            patterns.append(("acq",  re.compile(rf"acq-{re.escape(acq)}",  re.IGNORECASE)))
        if run:
            patterns.append(("run",  re.compile(rf"run-{re.escape(run)}",  re.IGNORECASE)))

        # Score each candidate by how many patterns it matches
        scored = []
        for fname in candidates:
            score = sum(bool(pat.search(fname)) for _, pat in patterns)
            scored.append((score, fname))

        # Pick highest score
        scored.sort(key=lambda x: ( -x[0], x[1].lower() ))  # descending score, then alpha
        best_score, best_fname = scored[0]

        # If it doesn’t even match the desc at all, treat as none found
        if best_score == 0:
            return None

        return join(dir_path, best_fname)
    
    def get_parcel_path(self,space,parc_scheme,scale,acq=None,run="01",grow=2,mode="chimera"):
        """""
        Returns the path to the Chimera or atlas based parcellation file.
        Args:
            parc_scheme : chimera parcellation scheme: LFMIHIFIF, LFMIHIFIS, cubic
            scale       : cortical parcellation scale or cube width if schema="cubic"
            space (str): Image space (e.g., "orig", "t1w", "mni").
            acq (str, optional): Acquisition parameter. Defaults to "memprage".
            run (str, optional): Run identifier. Defaults to "01".
            grow (int): GM growth into WM
        Returns:
            str : The matching file path.
        """""
        dirpath      = self.get_mri_parcel_dir_path("anat")
        _space="orig" if space.lower()=="t1w" or space.lower()=="anat" else space
        # prefix_name  = f"{self.prefix}_run-{run}_acq-{acq}_space-{space}_atlas-{parc_scheme}_dseg.nii.gz"
        if "cubic" in parc_scheme:
            prefix_name  = f"{self.prefix}_space-{_space}_atlas-{parc_scheme}{scale}mm_dseg.nii.gz"
            return join(dirpath,prefix_name) 
        else:
            if mode=="chimera":
                if   acq is not None and run is not None:
                    prefix_name  = f"{self.prefix}_run-{run}_acq-{acq}_space-{_space}_atlas-chimera{parc_scheme}_desc-scale{scale}grow{grow}mm_dseg.nii.gz"
                elif acq is not None and run is     None:
                    prefix_name  = f"{self.prefix}_acq-{acq}_space-{_space}_atlas-chimera{parc_scheme}_desc-scale{scale}grow{grow}mm_dseg.nii.gz"
                elif acq is None     and run is not None:
                    prefix_name  = f"{self.prefix}_run-{run}_space-{_space}_atlas-chimera{parc_scheme}_desc-scale{scale}grow{grow}mm_dseg.nii.gz"    
                elif acq is None     and run is     None:
                    prefix_name  = f"{self.prefix}_space-{_space}_atlas-chimera{parc_scheme}_desc-scale{scale}grow{grow}mm_dseg.nii.gz"    
                if grow is None:
                    prefix_name = prefix_name.replace("growNonemm","")        
                return join(dirpath,prefix_name) 
            elif mode=="atlas":
                if   acq is not None and run is not None:
                    prefix_name  = f"{self.prefix}_run-{run}_acq-{acq}_space-{_space}_atlas-{parc_scheme}_desc-scale{scale}_dseg.nii.gz"
                elif acq is not None and run is     None:
                    prefix_name  = f"{self.prefix}_acq-{acq}_space-{_space}_atlas-{parc_scheme}_desc-scale{scale}_dseg.nii.gz"
                elif acq is None     and run is not None:
                    prefix_name  = f"{self.prefix}_run-{run}_space-{_space}_atlas-{parc_scheme}_desc-scale{scale}_dseg.nii.gz"    
                elif acq is None     and run is     None:
                    prefix_name  = f"{self.prefix}_space-{_space}_atlas-{parc_scheme}_desc-scale{scale}_dseg.nii.gz"    

                return join(dirpath,prefix_name) 

    def get_mri_parcel_dir_path(self,modality="anat"):
        path = join(self.PARCEL_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            path = join(self.PARCEL_PATH.replace("-atlases",""),f"sub-{self.subject_id}",f"ses-{self.session}",modality)
            if os.path.exists(path):
                return path
            else:
                debug.warning("path does not exists")
                debug.warning(path)
                return 

    def get_parcel(self,space,parc_scheme,scale,acq="memprage",run="01",grow=2):
        path = self.get_parcel_path(space,parc_scheme,scale,acq,run,grow)
        if exists(path):
            return nib.load(path),path    
        else:
            debug.error("get_parcel: path does not exists",path)    
            raise("get_parcel: path does not exists",path)
 
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

    def get_connectivity_path(self,mode,parc_scheme,scale,npert=50,filtoption="",metabolite_suffix=None):
        dirpath     = self.get_connectivity_dir_path(mode)
        if mode=="mrsi":
            met_tag = ""
            if metabolite_suffix:
                if isinstance(metabolite_suffix, (list, tuple)):
                    tokens = [
                        re.sub(r"[^A-Za-z0-9]+", "", str(met)).lower()
                        for met in metabolite_suffix
                    ]
                    tokens = [tok for tok in tokens if tok]
                    suffix = "_".join(tokens)
                else:
                    suffix = re.sub(r"[^A-Za-z0-9_]+", "", str(metabolite_suffix)).strip("_")
                if suffix:
                    met_tag = f"_met-{suffix}"
            filename = f"{self.prefix}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{filtoption}{met_tag}_desc-connectivity_mrsi.npz"
        elif mode == "dwi": 
            filename = f"{self.prefix}_atlas-chimera{parc_scheme}_scale{scale}_desc-connectivity_dwi.npz"
        return join(dirpath,filename)

    def get_transform(self, direction, space):
        """
        Return the list of transform files for a given direction/space pair.

        Supported spaces:
            - "mrsi"                  :    MRSI -> T1w.
            - "anat"/"t1w"            :    T1w -> MNI (single visit).
            - "dwi"                   :    DWI -> T1w.
            - "t1-template"/"template":    T1w -> subject-specific template
                                          produced by registration_multivisit.sh.
            - "template-mni"/"ses-all":    Subject template (ses-all) -> MNI
                                          from registration_multivisit.sh.

        Args:
            direction (str): "forward" or "inverse".
            space (str): See list above (case-insensitive).
        """
        subject_label = f"sub-{self.subject_id}"
        session_label = f"ses-{self.session}"
        direction = direction.lower()
        space_key = space.lower()

        base_subject_dir = join(self.TRANSFORM_PATH, subject_label)

        def _build(space_dir_segments, prefix, has_affine_inverse=True):
            dir_path = join(*space_dir_segments)
            return dir_path, prefix, has_affine_inverse

        if space_key == "mrsi":
            transform_dir_path, transform_prefix, has_affine_inv = _build(
                (base_subject_dir, session_label, "mrsi"),
                f"{subject_label}_{session_label}_desc-mrsi_to_t1w",
            )
        elif space_key in {"t1w", "anat"}:
            transform_dir_path, transform_prefix, has_affine_inv = _build(
                (base_subject_dir, session_label, "anat"),
                f"{subject_label}_{session_label}_desc-t1w_to_mni",
            )
        elif space_key == "dwi":
            transform_dir_path, transform_prefix, has_affine_inv = _build(
                (base_subject_dir, session_label, "dwi"),
                f"{subject_label}_{session_label}_desc-dwi_to_t1w",
            )
        elif space_key in {"t1-template", "anat-template", "template"}:
            transform_dir_path, transform_prefix, has_affine_inv = _build(
                (base_subject_dir, session_label, "anat"),
                f"{subject_label}_{session_label}_desc-t1w_to_template",
                has_affine_inverse=False,
            )
        elif space_key in {"template-mni", "template_to_mni", "ses-all", "ses_all"}:
            transform_dir_path, transform_prefix, has_affine_inv = _build(
                (base_subject_dir, "ses-all", "anat"),
                f"{subject_label}_ses-all_desc-template_to_mni",
                has_affine_inverse=False,
            )
        else:
            raise ValueError(f"Unsupported transform space '{space}'.")

        if direction not in {"forward", "inverse"}:
            raise ValueError(f"Unsupported transform direction '{direction}'.")

        transform_list = []
        if direction == "forward":
            transform_list.append(join(transform_dir_path, f"{transform_prefix}.syn.nii.gz"))
            transform_list.append(join(transform_dir_path, f"{transform_prefix}.affine.mat"))
        else:
            affine_inv_path = join(transform_dir_path, f"{transform_prefix}.affine_inv.mat")
            if has_affine_inv or exists(affine_inv_path):
                transform_list.append(affine_inv_path)
            transform_list.append(join(transform_dir_path, f"{transform_prefix}.syn_inv.nii.gz"))
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
        - npert (int, optional): Number of perturbations.
        - filt (str, optional): Filter type.
        - met (str, optional): Metabolite tag.

        If any of these fields are not present, they are returned as None.
        """
        base_filename = os.path.basename(filename)
        results = {}

        # Define regex patterns for each field.
        patterns = {
            'sub':        r"sub-([^_]+)",
            'ses':        r"ses-([^_]+)",
            'run':        r"run-([^_]+)",
            'acq':        r"acq-([^_]+)",
            'space':      r"space-([^_]+)",
            'parcscheme': r"atlas-chimera([^_]+)",
            'scale':      r"scale(\d+)",
            'grow':       r"grow(\d+)mm",
            'npert':      r"npert(\d+)_",
            'filt':       r"filt([^_]+)",
            'met':        r"met-([^_]+)",   
            'res':        r"res-([^_]+)",   
        }

        for key, pat in patterns.items():
            m = re.search(pat, base_filename)
            results[key] = m.group(1) if m else None

        # Convert numeric fields to integers
        for numf in ('scale', 'grow', 'npert'):
            if results[numf] is not None:
                results[numf] = int(results[numf])

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
        debug.info("Looking for pattern",prefix,acq_patterns)
        # Recursively search for .nii files.
        for nifti_file in bids_dir.rglob("*.nii*"):
            # Check if the file name contains the subject/session pattern.
            if prefix not in nifti_file.name:
                continue
            debug.info("Candidate",nifti_file.name)
            if acq_patterns in nifti_file.name:
                return str(nifti_file)
        
        



if __name__=="__main__":
    mrsiData = MRIData(subject_id="CHUVA009",session="V5",group="LPN-Project")
    filepath = mrsiData.find_nifti_paths("desc-brain_T1w")
    print("Found",filepath)

    filename = "/media/veracrypt2/Connectome/Data/LPN-Project/derivatives/chimera-atlases/sub-CHUVA013_ses-V4_space-orig_res-5mm_met-GluGln_desc-signal_filtbiharmonic_mrsi.nii.gz"
    meta = mrsiData.extract_metadata(filename)
    # print(meta)

    t_list = mrsiData.get_transform( "forward","t1-template" )
    print(t_list)

    mrsiData = MRIData(subject_id="5741",session="V6",group="22q11-Project")
    pathlist = mrsiData.get_mri_filepath( "mrsi", "orig", "signal", met="GluGln")
    print(pathlist)                    

    nifti = mrsiData.get_mri_nifti( "mrsi", "orig", "signal", met="GluGln")
    print(nifti)
