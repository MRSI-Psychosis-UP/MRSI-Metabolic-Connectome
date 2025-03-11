import nibabel as nib
import numpy as np
import os
from os.path import join, split
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.stats import linregress
from scipy.stats import spearmanr
from registration.registration import Registration
from scipy.stats import chi2
from sklearn.metrics import mutual_info_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import random
from threading import Thread, Event
import csv, copy, sys, time
from tools.debug import Debug
from rich.progress import Progress,track
import scipy.stats as stats
import statsmodels.api as sm
from tools.datautils import DataUtils
from concurrent.futures import ProcessPoolExecutor
# from tqdm.auto import tqdm  # For progress bar
import itertools
from nilearn import datasets,image
from nilearn.image import resample_to_img
import xml.etree.ElementTree as ET
from scipy.stats import ConstantInputWarning
from nilearn.image import resample_img
from tools.filetools import FileTools

# Suppress only the ConstantInputWarning from scipy.stats
warnings.filterwarnings("ignore", category=ConstantInputWarning)


dutils = DataUtils()
debug  = Debug()
reg    = Registration()
ftools   = FileTools()






class Parcellate:
    def __init__(self) -> None:
        # self.PARCEL_PATH = "/media/flucchetti/77FF-B8071/Mindfulness-Project/derivatives/chimera-atlases/"
        self.PARCEL_PATH      = join(dutils.BIDSDATAPATH,"Mindfulness-Project","derivatives","chimera-atlases")
        self.PARCEL_PATH_ARMS = join(dutils.BIDSDATAPATH,"LPN-Project","derivatives","chimera-atlases")
        self.mni_template     = datasets.load_mni152_template()
        self.mni_mask         = datasets.load_mni152_brain_mask()
        self.mni_gm_mask      = datasets.load_mni152_gm_mask()
        pass

    def create_parcel_image(self,atlas_string="aal"):
        # Load the AAL atlas
        if atlas_string == "aal":
            atlas   = datasets.fetch_atlas_aal(version='SPM12')
            labels  = atlas['labels']
            indices = atlas['indices']
            maps    = atlas['maps']
            header  = nib.load(maps).header
            start_idx = 1
            # Load the T1 image
            # Resample the atlas to the T1 image
            parcel_image = nib.load(maps).get_fdata()
            # parcel_image = nib.load(maps).get_fdata()
        elif atlas_string== "destrieux":
            atlas = datasets.fetch_atlas_destrieux_2009()
            labels,indices = list(),list()
            for entry in atlas['labels']:
                if entry[0]==0:continue
                indices.append(entry[0])
                labels.append(entry[1])
            maps   = atlas['maps']
            header = nib.load(maps).header
            start_idx = 1
            parcel_image = nib.load(maps).get_fdata()
        elif atlas_string == "jhu_icbm_wm":
            maps = join(dutils.DEVANALYSEPATH,"data","atlas","jhu_icbm_wm","JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz")
            indices, labels = self.parse_atlas_xml(maps.replace(".nii.gz",".xml"))
            indices=np.array(indices).astype(int)+1 # starts at 0
            header = nib.load(maps).header
            start_idx = 8001
            parcel_image = nib.load(maps).get_fdata()
        elif atlas_string == "mist-197":
            mist_atlas      = datasets.fetch_atlas_basc_multiscale_2015()
            parcel_image_og = mist_atlas.scale197
            parcel_image_ni = image.resample_to_img(parcel_image_og, self.mni_template, interpolation='nearest')
            parcel_image    = parcel_image_ni.get_fdata()
            header          = parcel_image_ni.header
            indices, labels = self.read_roi_label(join(dutils.DEVANALYSEPATH,"data","atlas","mist197","MIST_197.csv"))
            indices         = np.array(indices).astype(int)
            return parcel_image.astype(int), labels, indices, header

        elif atlas_string == "schaefer-200":
            atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
            # atlas_schaefer.labels = np.insert(atlas_schaefer.labels, 0, "Background")
            parcel_image_og = atlas_schaefer['maps'] 
            labels          = atlas_schaefer['labels'] 
            parcel_image_ni = image.resample_to_img(parcel_image_og, self.mni_template, interpolation='nearest')
            parcel_image    = parcel_image_ni.get_fdata()
            header          = parcel_image_ni.header
            indices         = np.arange(1,parcel_image.max())
            indices         = np.unique(parcel_image)
            return parcel_image.astype(int), labels, indices[indices!=0], header
            
        elif atlas_string == "cerebellum":
            maps = join(dutils.DEVANALYSEPATH,"data","atlas","cerebellum_mnifnirt","Cerebellum_MNIfnirt.nii.gz")
            indices, labels = self.parse_atlas_xml(maps.replace(".nii.gz",".xml"),prefix="cer-")
            indices         = np.array(indices).astype(int)+1 # starts at 0
            header          = nib.load(maps).header
            start_idx       = 2001
            parcel_image    = nib.load(maps)
            reference_img   = datasets.load_mni152_template()

            parcel_image    = resample_img(parcel_image, 
                                           target_affine=reference_img.affine, 
                                           target_shape=reference_img.shape,
                                           interpolation="nearest").get_fdata()

        elif atlas_string == "geometric_cubeK23mm":
            gm_mask_mni152  = datasets.load_mni152_gm_mask()
            parcel_image    = self.parcellate_volume(gm_mask_mni152.get_fdata(), K=23)
            indices         = np.arange(1,parcel_image.max())
            header          = gm_mask_mni152.header
            start_idx       = 1
            labels          = [f"gm-{index}" for index in indices]
        elif atlas_string == "geometric_cubeK23mm":
            gm_mask_mni152  = datasets.load_mni152_gm_mask()
            parcel_image    = self.parcellate_volume(gm_mask_mni152.get_fdata(), K=23)
            indices         = np.arange(1,parcel_image.max())
            header          = gm_mask_mni152.header
            start_idx       = 1
            labels          = [f"gm-{index}" for index in indices]
        elif atlas_string == "geometric_cubeK18mm":
            gm_mask_mni152  = datasets.load_mni152_gm_mask()
            parcel_image    = self.parcellate_volume(gm_mask_mni152.get_fdata(), K=18)
            indices         = np.arange(1,parcel_image.max())
            header          = gm_mask_mni152.header
            start_idx       = 1
            labels          = [f"gm-{index}" for index in indices]
        elif atlas_string == "wm_cubeK18mm":
            start_idx       = 9001
            wm_mask_mni152  = datasets.load_mni152_wm_mask()
            gm_mask_mni152  = datasets.load_mni152_gm_mask()
            wm_mask         = copy.deepcopy(wm_mask_mni152.get_fdata())
            wm_mask[gm_mask_mni152.get_fdata()==1] = 0
            parcel_image    = self.parcellate_volume(wm_mask, K=18)
            indices         = np.arange(1,parcel_image.max())
            header          = wm_mask_mni152.header
            labels          = [f"wm-{index}" for index in indices]
        else:
            debug.error(atlas_string,"not recognized")
            return


        # Create a mapping of indices to labels
        parcel_image     = parcel_image.astype(int)
        new_indices      = np.arange(start_idx,start_idx+len(indices)).astype(int)
        new_parcel_image = np.zeros(parcel_image.shape)
        for i, index in enumerate(indices):
            mask = parcel_image == index
            new_parcel_image[mask] = new_indices[i]

        parcel_image +=start_idx
        if "wm_cubeK" not in atlas_string:
            new_parcel_image[~self.mni_gm_mask.get_fdata().astype(bool)] = 0
        return new_parcel_image.astype(int), labels, new_indices, header

    def generate_random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def create_tsv(self,labels,indices, output_file):
        with open(output_file, 'w') as f:
            f.write("index\tname\tcolor\n")
            for index, label in zip(indices,labels):
                # index = indices[i]
                color = self.generate_random_color()
                f.write(f"{index}\t{label}\t{color}\n")


    def merge_gm_wm_parcel(self,gmParcel, wmParcel):
        return np.where(gmParcel != 0, gmParcel, wmParcel)


    def merge_gm_wm_dict(self,parcel_header_dict_gm,parcel_header_dict_wm):
        parcel_header_dict = copy.deepcopy(parcel_header_dict_gm)
        parcel_header_dict.update(parcel_header_dict_wm)   # Update the copy with the second dictionary
        return parcel_header_dict

    def get_parcel_path(self,subject_id):
        debug.info("get_parcel_path for",subject_id)
        filename = "run-01_acq-memprage_space-orig_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg.tsv"
        # filename = "run-01_acq-memprage_space-orig_atlas-chimeraLFIIHIFIF_desc-scale1grow2mm_dseg.tsv"
        S,V = subject_id[0][0:4],subject_id[0][5:]
        if V=="V2_BIS":V="V3"
        path = join(self.PARCEL_PATH,f"sub-{S}", 
                                  f"ses-{V}", 
                                  "anat", 
                                  f"sub-{S}_ses-{V}_{filename}")
    
        return path


    def get_parcel_header(self,parcel_header_path,cutoff=None,ignore_parcel_list=[]):
        '''
        Retrieves and filters a brain parcel image for a given subject 
        based on an ignore list. 

        Parameters:
        - parcel_header_path (str): Unique identifier for the subject whose parcel data is to be retrieved.
        - cutoff: vut off parcel labels above value default=3000 [WM parcels]
        Returns:
        - tuple: 
            - filtered parcel image (numpy array), 
            - filtered list of parcel IDs, 
            - filtered list of labels (where each label is a list split by '-'), 
            - filtered list of color codes. 
        '''
        parcel_ids, label_list, color_codes = self.read_tsv_file(parcel_header_path,ignore_parcel_list)
        label_list = [label.split('-') for label in label_list]
        
        header_dict = dict()
        for idx , parcel_id in enumerate(parcel_ids):
            if cutoff is None or parcel_id < cutoff:
                header_dict[int(parcel_id)] = {"label":label_list[idx],"color":color_codes[idx],"mask":1,
                                                "count":[],"mean":0,"std":0,"med":0,"t1cov":[]}
        header_dict[0] = {"label":["BND"],"color":0,"mask":0,"count":[0],"mean":0,"std":0,"med":0,"t1cov":[0]}

        return header_dict


    def tranform_parcel(self,target_image_path,parcel_path,transform):     
        parcel_image3D           = reg.transform(target_image_path,
                                                parcel_path,
                                                transform,
                                                interpolator_mode="genericLabel")
        return parcel_image3D.numpy()

    def filter_parcel(self,parcel_image,parcel_header_dict ,ignore_list=[]):
        '''
        Filters a given parcel image and its corresponding 
        metadata (parcel IDs, labels, and color codes) by removing 
        parcels specified in an ignore list. 

        Parameters:
        - parcel_image (numpy array): The brain parcel image 
        - parcel_ids (list of int): List of unique parcel IDs 
        - label_list (list of list of str): List of parcel labels 
        - color_codes (list of str): List of color codes 
        - ignore_list (list of str): Labels of parcels to be ignored

        Returns:
        - tuple: A tuple containing a  filtered version of the input
        '''
        filt_parcel_image = copy.deepcopy(parcel_image)


        # Iterate over a copy of the keys list to safely modify the dictionary
        for label_idx in list(parcel_header_dict.keys()):
            entry = parcel_header_dict[label_idx]
            subparcel_labels = entry["label"]
            for subparcel_label in subparcel_labels:
                if subparcel_label in ignore_list:
                    filt_parcel_image[parcel_image == label_idx] = 0
                    parcel_header_dict[label_idx]["mask"] = 0
                    del parcel_header_dict[label_idx]
                    break 
            else:
                parcel_header_dict[label_idx]["mask"] = 1

        return filt_parcel_image,parcel_header_dict

    def merge_parcels(self,parcel_image,parcel_header_dict, merge_parcels_dict):
        # debug.error("parcel_header_dict[28]",parcel_header_dict[28])
        merged_parcel_image = copy.deepcopy(parcel_image)
        for key, entry in merge_parcels_dict.items():
            base_label = entry["merge"][0]
            merge_ids  = range(entry["merge"][0] + 1, entry["merge"][1] + 1)  # exclude parcel id to merge with
            # debug.success("parcel_header_dict[key]",parcel_header_dict[int(key)])
            # debug.info(key, entry,entry["label"])
            parcel_header_dict[int(key)]["label"] = entry["label"]
            for merge_idx in merge_ids:
                # Update the parcel image to merge the labels
                merged_parcel_image[parcel_image == merge_idx] = base_label
                # Delete the merged label's entry from the header dict if it exists
                parcel_header_dict.pop(merge_idx, None)
        return merged_parcel_image, parcel_header_dict

    
    def read_tsv_file(self,filepath,ignore_parcel_list=[]):
        """
        Reads a TSV file with three columns: an integer, a string label, and a color code.
        Returns the data as three separate lists.

        Parameters:
        - filepath: Path to the TSV file.

        Returns:
        - numbers: List of integers from the first column.
        - labels: List of string labels from the second column.
        - colors: List of color codes (strings) from the third column.
        """
        numbers, labels, colors = [], [], []

        with open(filepath, 'r') as file:
            tsv_reader = csv.reader(file, delimiter='\t')
            next(tsv_reader, None)  # Skip the header row
            for row in tsv_reader:
                # Assuming the file structure is exactly as described
                number, label, color = row
                if any(ignore_parcel in label for ignore_parcel in ignore_parcel_list):
                    continue
                numbers.append(int(number))  # Convert string to int
                labels.append(label)
                colors.append(color)

        return np.array(numbers), labels, colors


    def get_parcellation_image(self,filepath,ignore_parcel_list=[]):
        parc_image         = nib.load(filepath)
        parc_image_np      = parc_image.get_fdata().astype(int)
        indices, _, _ = self.read_tsv_file(filepath.replace(".nii.gz",".tsv"),ignore_parcel_list)
        filtered_parcel_image = np.zeros(parc_image_np.shape)
        for parcel_idx in indices:
            filtered_parcel_image[parc_image_np==parcel_idx] = parcel_idx
        return ftools.numpy_to_nifti(filtered_parcel_image,parc_image.header)
         

    def count_voxels_per_parcel(self,parcel_image, mask_mrsi,mask_t1,parcel_header_dict):
        """
        Identifies parcel IDs that are completely ignored based on the mask, 
        meaning there are fewer than 'threshold' voxels available for that specific parcel.
        Parameters:
        - parcel_image: A 3D numpy array with parcel IDs.
        - mask: A 3D numpy array with 0s for background and 1s for foreground.
        - threshold: The minimum number of voxels required to not ignore a parcel (default is 10).
        Returns:
        - ignored_parcels: A list of parcel IDs that are ignored based on the mask.
        """
        unique_parcels = np.array(list(parcel_header_dict.keys()))
        for parcel_id in unique_parcels:
            # Count voxels for the current parcel_id that are also in the foreground
            voxel_count_mrsi = np.sum((parcel_image == parcel_id) & (mask_mrsi == 1))
            parcel_header_dict[parcel_id]["count"].append(voxel_count_mrsi)
            voxel_count_t1 = np.sum((parcel_image == parcel_id) & (mask_t1 == 1))
            parcel_header_dict[parcel_id]["t1cov"].append(voxel_count_mrsi/voxel_count_t1)
        return  parcel_header_dict



    
    def count_voxels_inside_parcel(self,image3D, parcel_image3D, parcel_ids_list,norm=True):
        parcel_count = {}

        # Loop through each parcel ID in the provided list
        for parcel_id in parcel_ids_list:
            # Skip if the parcel ID is 0
            if parcel_id == 0:
                continue
            
            # Create a mask for the current parcel ID
            parcel_mask = parcel_image3D == parcel_id
            
            # Check if there are any voxels in this parcel
            n_total = parcel_mask.sum()
            if n_total == 0:
                continue
            
            # Count the number of voxels in the image3D that are inside the current parcel
            image_mask = image3D[parcel_mask].sum()
            

            # Calculate the percentage coverage
            if norm:
                n_coverage = image_mask / n_total
            else:
                n_coverage = image_mask
            
            # Assign the count to the dictionary
            parcel_count[parcel_id] = n_coverage
        
        # Return the dictionary containing the voxel counts
        return parcel_count
   


    def normalizeMetabolites4D(self,met_image4D_data):
        normalized_4d = np.zeros(met_image4D_data.shape)
        for i, image3D in enumerate(met_image4D_data):
            normalized_4d[i] = image3D/np.max(image3D)
        return normalized_4d
    
    def initialize_label_dict(self,gm_mask_np):
        """
        Initializes a dictionary for the given grey matter mask numpy array.

        Parameters:
        - gm_mask_np: 3D numpy array representing the grey matter mask.

        Returns:
        - label_dict: Dictionary with the specified structure.
        """
        unique_labels = np.unique(gm_mask_np)
        label_dict = {}

        for key in range(len(unique_labels)):
            label_dict[key] = {
                'label': [key],
                'color': self.generate_random_color(),
                'mask': 1,
                'count': [],
                'mean': 0,
                'std': 0,
                'med': 0,
                't1cov': []
            }
        
        return label_dict
