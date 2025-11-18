import numpy as np
import nibabel as nib
import math
from scipy.ndimage import median_filter
from skimage.restoration import inpaint_biharmonic
from nilearn import image as nil_img
from scipy.ndimage import generic_filter

from tools.filetools import FileTools


ftools     = FileTools()

class BiHarmonic:
    def __init__(self):
        """
        Initialize the BiHarmonic processing object.
        """
        pass

    def proc(self, image_og, brain_mask, fwhm=8, channel=0,percentile=95):
        """
        Process the input image by detecting spikes, inpainting defects, and applying spatial filtering.
        
        Parameters:
            image_og (nib.Nifti1Image): The original image data. 
            brain_mask (np.ndarray): Boolean mask to specify the brain region. Voxels outside this mask will be set to zero.
            fwhm (float, optional): Full width at half maximum (FWHM) in mm for the spatial filter. If None, it is computed
                                    from the header voxel dimensions. Default is 8.
            channel (int, optional): Channel index to use if image_og is a 4D array. Default is 0.
        
        Returns:
            np.ndarray: The processed image data after spike removal, inpainting, and spatial filtering.
        """
        # If input is a Nifti image, extract data and header.
        if isinstance(image_og, nib.Nifti1Image):
            image_og_np = image_og.get_fdata()
            header      = image_og.header
        # If input is a NumPy array, use it directly. If 4D, select the specified channel.
        else:
            raise TypeError("image_og must be a nib.Nifti1Image")
        if image_og_np.ndim == 4:
            image_og_np = image_og_np.squeeze()  

        # If input is a Nifti image, extract data and header.
        if isinstance(brain_mask, nib.Nifti1Image):
            brain_mask_np = brain_mask.get_fdata().astype(bool)
        # If input is a NumPy array, use it directly. If 4D, select the specified channel.
        else:
            raise TypeError("brain_mask must be a nib.Nifti1Image")
        if brain_mask_np.ndim == 4:
            brain_mask_np = brain_mask_np[:, :, :].squeeze().astype(bool) 


        
        # Detect spikes based on a sigma threshold converted to a percentile.
        spike_mask    = self.get_spike_mask(image_og_np, percentile=percentile, bnd_np=None)
        image_unspiked_np   = self.inpaint_voxels_with_median(image_og_np, spike_mask)
        # Detect holes (NaNs) in the image.
        nan_mask = np.isnan(image_unspiked_np)
        # Detect missing values in image wrt brain mask in the image.
        missing_mask = np.zeros_like(image_unspiked_np).astype(bool)
        missing_mask[(image_unspiked_np==0) & (brain_mask_np==1)]=True
        # Combine masks for inpainting.
        inpaint_mask = (nan_mask | spike_mask) | missing_mask
        # Inpaint the detected defects using biharmonic inpainting.
        image_inpaint_np = self.biharmonic(image_unspiked_np, mask=missing_mask)
        image_inpaint_np = self.inpaint_voxels_with_median(image_unspiked_np, missing_mask)
        
        # If fwhm is None and header information is available, compute fwhm based on voxel dimensions.
        if fwhm is None and header is not None:
            voxel_dims = np.array(header.get_zooms()[:3])
            fwhm = np.round(voxel_dims.mean() * np.sqrt(2))
        
        # Apply spatial filtering (e.g., smoothing) to the inpainted image.
        image_inpaint_nifti = ftools.numpy_to_nifti(image_inpaint_np,header)
        image_smoothed_np = self.spatial_filter(image_inpaint_nifti, fwhm=fwhm, mask=inpaint_mask)
        # Zero out voxels outside the brain mask.
        image_smoothed_np[~brain_mask_np] = 0
        return ftools.numpy_to_nifti(image_smoothed_np,header)


    @staticmethod
    def get_spike_mask(data_np, percentile=95, bnd_np=None):
        """
        Generate a boolean mask for spike detection based on a sigma threshold converted to a percentile.
        
        Parameters:
            data_np (np.ndarray): The image data.
            sigma (float): The sigma value to convert into a percentile threshold (default is 3.5).
            bnd_np (np.ndarray, optional): A boolean mask to restrict the data region. Defaults to data > 0.
        
        Returns:
            np.ndarray: A boolean mask where True indicates a spike.
        """
        # Compute the one-sided percentile corresponding to the given sigma value.
        # Multiply by 100 because np.percentile expects a value in the range [0, 100].
       
        # Default to a mask where data is positive if no boundary mask is provided.
        if bnd_np is None:
            bnd_np = data_np > 0
        
        # Compute the threshold value at the given percentile.
        threshold_val = np.percentile(data_np[bnd_np], percentile)
        # Create the spike mask: True for values exceeding the threshold.
        spike_mask = data_np > threshold_val
        
        return spike_mask

    @staticmethod
    def biharmonic(image_orig, mask):
        """
        Apply biharmonic inpainting to repair defects in the image.
        
        Parameters:
            image_orig (np.ndarray): The original image data.
            mask (np.ndarray): Boolean mask indicating defective regions (True where defects exist).
        
        Returns:
            np.ndarray: The inpainted image.
        """
        # Create an image with defects removed (set to 0 outside the mask).
        image_defect = image_orig.copy()
        image_defect[mask] = 0
        # Inpaint the defects using biharmonic inpainting.
        image_result = inpaint_biharmonic(image_defect, mask)
        return image_result

    @staticmethod
    def spatial_filter(image, fwhm=5, mask=None, filter_size=3, channel=0):
        """
        Apply a spatial filter to the image by replacing defective voxels with local median values,
        followed by smoothing.
        
        Parameters:
            image_np (np.ndarray): The image data (3D NumPy array).
            fwhm (float, optional): Full width at half maximum for the smoothing filter in mm.
            mask (np.ndarray, optional): Boolean mask indicating voxels to be replaced. If provided,
                                           those voxels will be set to the local median.
            filter_size (int, optional): The size of the neighborhood for the median filter. Default is 3.
        
        Returns:
            np.ndarray: The spatially filtered (smoothed) image data.
        """
        if isinstance(image, nib.Nifti1Image):
            image_og_np = image.get_fdata()
            header = image.header
        # If input is a NumPy array, use it directly. If 4D, select the specified channel.
        elif isinstance(image, np.ndarray):
            image_og_np = image
            if image_og_np.ndim == 4:
                image_og_np = image_og_np[:, :, :, channel]
            header = None  # No header available when image_og is a NumPy array.
        else:
            raise TypeError("image_og must be a nib.Nifti1Image or a NumPy array.")

        # # Compute the local median using a median filter.
        # local_median = median_filter(image_og_np, size=filter_size)
        # # Replace voxels specified in the mask with the local median values.
        # if mask is not None:
        #     image_og_np = image_og_np.copy()
        #     image_og_np[mask] = local_median[mask]
        
        # Create a Nifti image for smoothing. If no header/affine is provided, assume identity affine.
        nifti_img = ftools.numpy_to_nifti(image_og_np,header)
        # Apply spatial smoothing using nilearn's smooth_img function.
        smoothed_img = nil_img.smooth_img(nifti_img, fwhm=fwhm)
        # Extract the smoothed data as a NumPy array.
        image_smoothed_np = smoothed_img.get_fdata()  
        image_og_np[mask] = image_smoothed_np[mask]    
        return image_smoothed_np

    @staticmethod
    def create_surrounded_holes_mask(mask_img):
        """
        Create a binary mask of voxels that are 0 and are completely surrounded by voxels with a value of 1.
        This function is intended for binary brain masks where 1 indicates the brain and 0 indicates missing values.
        
        Parameters:
            mask_img (np.ndarray): A 3D numpy array representing the binary brain mask.
            
        Returns:
            np.ndarray: A binary (bool) 3D mask where True indicates that the voxel is 0 (a hole)
                        and all its 6-connected neighbors (in the x, y, z directions) are 1.
                        
        Note:
            Boundary voxels are not considered since they do not have a full set of 6 neighbors.
        """
        # Ensure the input is a numpy array.
        mask_img = np.asarray(mask_img)
        
        # Create boolean arrays for holes (0) and valid brain (1) voxels.
        hole_voxels = (mask_img == 0)
        valid_voxels = (mask_img == 1)
        
        # Initialize the output mask with False.
        surrounded_holes = np.zeros_like(mask_img, dtype=bool)
        
        # For interior voxels, check if a voxel is a hole and all its 6 orthogonal neighbors are valid.
        surrounded_holes[1:-1, 1:-1, 1:-1] = (
            hole_voxels[1:-1, 1:-1, 1:-1] &
            valid_voxels[2:  , 1:-1, 1:-1] &  # neighbor in +x
            valid_voxels[:-2, 1:-1, 1:-1] &  # neighbor in -x
            valid_voxels[1:-1, 2:  , 1:-1] &  # neighbor in +y
            valid_voxels[1:-1, :-2, 1:-1] &  # neighbor in -y
            valid_voxels[1:-1, 1:-1, 2:  ] &  # neighbor in +z
            valid_voxels[1:-1, 1:-1, :-2]    # neighbor in -z
        )
        brain_mask_restored = mask_img | surrounded_holes
        return brain_mask_restored, surrounded_holes

    @staticmethod
    def __median_exclude_center(values):
        """
        Given a flattened neighborhood (e.g., from a 3x3x3 block),
        compute the median of the surrounding voxels while excluding the center voxel.
        """
        center_index = len(values) // 2  # For a 3x3x3, index 13 in a flattened 27-element array.
        # Exclude the center element.
        neighbors = np.concatenate((values[:center_index], values[center_index+1:]))
        return np.median(neighbors)

    def inpaint_voxels_with_median(self,image, binary_mask, filter_size=3):
        """
        For voxels where binary_mask is True, replace the image value with the median
        of the surrounding voxels (excluding the voxel itself).

        Args:
            image (numpy.ndarray): A 3D NumPy array representing the brain image.
            binary_mask (numpy.ndarray): A boolean mask of the same shape as image where True indicates voxels to filter.
            filter_size (int): Size of the neighborhood to compute the median (default is 3 for a 3x3x3 cube).

        Returns:
            numpy.ndarray: A new 3D image with the specified voxels replaced.
        """
        # Compute a median-filtered image for every voxel using the custom function.
        # The mode 'mirror' handles border voxels gracefully.
        median_image = generic_filter(image, self.__median_exclude_center, size=filter_size, mode='mirror')
        
        # Copy the original image so that we only update the specified voxels.
        filtered_image = image.copy()
        filtered_image[binary_mask] = median_image[binary_mask]
        return filtered_image
