

import numpy as np
import nibabel as nib
from nilearn import datasets, image
from tools.filetools import FileTools


ftools = FileTools()
# Load source template (1 mm MNI)
# mni_5mm = datasets.load_mni152_template(resolution=2)
# ftools.save_nii_file(mni_5mm,"mni_1mm.nii.gz")


class ImageTools(object):
    def __init__(self):
        pass

    def get_mni(res=1):
        return datasets.load_mni152_template(resolution=res)




