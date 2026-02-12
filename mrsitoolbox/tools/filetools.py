
import os,sys, glob,re, shutil,json

from mrsitoolbox.tools.datautils import DataUtils
from mrsitoolbox.tools.debug import Debug 
import nibabel as nib
import numpy as np
from os.path import join, split
dutils = DataUtils()
debug  = Debug()
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class FileTools:
    def __init__(self) -> None:
       pass
    
    def save_nii_file(self, image, outpath,header=None):
        """
        Save an image to a NIfTI file.

        Parameters:
        image: A nibabel Nifti1Image or a numpy array.
        header: The NIfTI header. Must be provided if image is a numpy array.
        outpath: The file path to save the image.
        """
        if isinstance(image, nib.Nifti1Image):
            nifti_img = image
        elif isinstance(image, np.ndarray):
            if header is not None:
                nifti_img = self.numpy_to_nifti(image, header)
            else:
                raise ValueError("Provide header for saving np.ndarray to NIFTI")
        else:
            raise ValueError("Input must be a Nifti1Image or a numpy array")
        
        nifti_img.to_filename(outpath)

    @staticmethod
    def save_dict(python_dict,outpath):
        with open(outpath, 'w') as f:
            json.dump(python_dict, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    @staticmethod
    def numpy_to_nifti(tensor, header):
        """
        Convert a numpy array to a NIfTI image reusing metadata from an existing header.

        The original header is copied so that spatial metadata (affine, zooms, etc.)
        are preserved, and its dimensionality is updated to match the new tensor.
        """
        if header is None:
            raise ValueError("Header must be provided to convert numpy array to NIfTI.")

        affine = header.get_best_affine()
        new_header = header.copy()
        new_header.set_data_dtype(np.float32)
        new_header.set_data_shape(tensor.shape)
        nifti_img = nib.Nifti1Image(tensor.astype(np.float32), affine, header=new_header)
        return nifti_img

    def save_transform(self,transform,dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Copy each file in the transform list to the custom directory
        for transform_file in transform:
            filename = split(transform_file)[1]
            if "Warp" in filename:
                filename="Warp.nii.gz"
            elif "Affine" in filename:
                filename="GenericAffine.mat"
            dest_file_path = join(dir_path, filename)
            # Copy the file to the new location
            shutil.copy(transform_file, dest_file_path)
            debug.success(f"Saved: {filename}")
 
    def save_nii4D_file(self,path_list,outpath):
        _data = nib.load(path_list[0])
        image_list=np.zeros([_data.get_fdata().shape+(len(path_list),)])
        header = _data.header
        for ids,path in enumerate(path_list):
            data = nib.load(path)
            image_list[:,:,:,ids] = data.get_fdata()
        self.save_nii_file(np.array(image_list),header,f"{outpath}.nii.gz")

    def list_nii_files(self,directory):
        """
        Lists the absolute paths of all .nii files within the given directory, including its subdirectories.
        
        Parameters:
        - directory: The root directory to search for .nii files.
        
        Returns:
        - A list of absolute paths to each .nii file found.
        """
        nii_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".nii") and not file.endswith(".nii.gz"):
                    nii_files.append(os.path.abspath(os.path.join(root, file)))
        return nii_files

    @staticmethod
    def list_recordings(group,modality="mrsi"):
        bidsdir = join(dutils.BIDSDATAPATH,group)
        recording_list = list()
        subject_list = os.listdir(bidsdir)
        for subject_id in subject_list:
            if "sub-" not in subject_id:continue
            session_list = os.listdir(join(bidsdir,subject_id))
            
            for session in session_list:
                if "ses-" in session:
                    print("subject_list",subject_id,session)
                    acq_list = os.listdir(join(bidsdir,subject_id,session))
                    mrsi_dir_path = join(bidsdir,subject_id,session,modality)
                    if os.path.exists(mrsi_dir_path):
                        n_mrsi = len(os.listdir(mrsi_dir_path))
                        if n_mrsi!=0:
                            recording_list.append([subject_id[4::],session[4::]])
        recording_list = np.array(recording_list)
        print(recording_list)
        ids = np.argsort(recording_list[:,0])
        return recording_list[ids,:]





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
                if len(row)!=3:continue
                number, label, color = row
                if any(ignore_parcel in label for ignore_parcel in ignore_parcel_list):
                    continue
                numbers.append(int(number))  # Convert string to int
                labels.append(label)
                colors.append(color)
        return np.array(numbers), labels, colors





if __name__=="__main__":
    ft = FileTools()
    debug.info(ft.list_mrsi_subject_list())
