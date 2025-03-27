
import os,sys, glob,re, shutil,json

from tools.datautils import DataUtils
from tools.debug import Debug 
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
    
    def save_nii_file(self, tensor3D, header,outpath):
        nifti_img = self.numpy_to_nifti(tensor3D, header)
        nifti_img.to_filename(f"{outpath}")

    @staticmethod
    def save_dict(python_dict,outpath):
        with open(outpath, 'w') as f:
            json.dump(python_dict, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    @staticmethod
    def numpy_to_nifti(tensor3D, header):
        affine = header.get_best_affine()
        # Preserve affine transform
        header.set_data_dtype(np.float32)
        nifti_img = nib.Nifti1Image(tensor3D.astype(np.float32), affine)
        
        # Update specific fields in the new header from the original header
        for key in header.keys():
            try:
                nifti_img.header[key] = header[key]
            except Exception as e:
                debug.warning(f"Could not set header field '{key}': {e}")
        
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
