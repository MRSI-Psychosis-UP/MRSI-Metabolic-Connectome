
import numpy as np
from mrsitoolbox.tools.debug import Debug

# Prefer CuPy, but fall back to NumPy if the CUDA toolchain is unavailable or misconfigured.
try:
    import cupy as _cupy
    _cupy.asarray(np.zeros(1))  # basic check
    if _cupy.cuda.runtime.getDeviceCount() > 0:
        try:
            # Trigger a simple elementwise kernel; fails fast if CUDA headers/toolkit missing.
            _ = (_cupy.asarray(np.array([1.0], dtype=np.float32)) * 2).sum()
            cp = _cupy
            USE_GPU = True
        except Exception:
            cp = np
            USE_GPU = False
    else:
        raise RuntimeError("No GPU detected.")
except Exception:
    cp = np
    USE_GPU = False

import nibabel as nib
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA




debug  = Debug()


class NetTools:
    def __init__(self) -> None:
        pass


    def compute_centroids(self, label_image, labels=None, world=False):
        """
        Computes centroids for all regions in a label image.

        Parameters:
        label_image: NIfTI image or numpy array of labels.
        labels: iterable of label ids (defaults to unique labels in the image).
        world: if True, return coordinates in world (affine) space for NIfTI inputs.
        """
        img = None
        if hasattr(label_image, "get_fdata"):
            img = label_image
            data = label_image.get_fdata()
        else:
            data = np.asarray(label_image)

        if labels is None:
            labels = np.unique(data)

        # Use center_of_mass with `index` argument to calculate all centroids at once
        centroids = np.array(center_of_mass(data, labels=data, index=labels))

        if world:
            if img is None:
                raise ValueError("world=True requires a NIfTI image with an affine.")
            centroids = nib.affines.apply_affine(img.affine, centroids)

        return centroids


    @staticmethod
    def compute_parcel_centers(image_nifti):
        """
        Computes the average (x, y, z) geometric center coordinates of each parcel in an MNI space parcel label image.

        Parameters:
        nifti_file_path (str): Path to the NIfTI file containing the parcel label image.

        Returns:
        dict: A dictionary where keys are parcel labels and values are tuples of (x, y, z) coordinates.
        """
        # Load the data from the NIfTI image
        data = image_nifti.get_fdata()
        # Get the affine transformation matrix to map voxel indices to MNI space
        affine = image_nifti.affine
        # Get unique labels (parcels) in the image, excluding the background label 0
        labels = np.unique(data)
        labels = labels[labels != 0]  # Exclude background (assuming 0 is background)
        # Initialize arrays to store coordinates sum and density
        parcel_centers = {}
        coord_sum = np.zeros((len(labels), 3))
        voxel_count = np.zeros(len(labels))
        # Create a mapping of labels to indices in the array for faster access
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        # Iterate over all voxels to accumulate sums and density per label
        for index, value in np.ndenumerate(data):
            if value in label_to_index:
                idx = label_to_index[value]
                coord_sum[idx] += np.array(index)  # Accumulate voxel indices
                voxel_count[idx] += 1             # Count the voxel
        # Compute the mean voxel coordinate and transform to world coordinates
        for label, idx in label_to_index.items():
            center = coord_sum[idx] / voxel_count[idx]  # Mean voxel coordinate
            world_center = nib.affines.apply_affine(affine, center)  # Convert to MNI space
            parcel_centers[int(label)] = tuple(world_center)
        return parcel_centers

    def compute_distance_matrix(self,centroids):   
        distance_matrix = cdist(centroids, centroids, metric='euclidean')
        return distance_matrix

   

    @staticmethod
    def project_to_3dspace(feature_arr, parcellation_data_np, label_indices):
        """
        Map 1D nodal features back to a 3D volume.
        Uses GPU via CuPy when available; otherwise falls back to NumPy.
        """
        def _fill_map(xp_backend):
            parcellation_backend = xp_backend.asarray(parcellation_data_np)
            nodal_strength_map = xp_backend.zeros(parcellation_backend.shape)
            label_to_similarity = {
                label: similarity for label, similarity in zip(label_indices, feature_arr)
            }
            for label, similarity in label_to_similarity.items():
                nodal_strength_map[parcellation_backend == label] = similarity
            nodal_strength_map[parcellation_backend == 0] = 0
            return nodal_strength_map

        try:
            if USE_GPU:
                nodal_strength_map = _fill_map(cp)
                return cp.asnumpy(nodal_strength_map)
            nodal_strength_map = _fill_map(np)
            return np.asarray(nodal_strength_map)
        except Exception as exc:
            if USE_GPU:
                debug.warning("GPU projection failed; retrying on CPU", exc)
                nodal_strength_map = _fill_map(np)
                return np.asarray(nodal_strength_map)
            raise



    def dimreduce_matrix(self, data, method='pca_tsne', scale_factor=255.0, output_dim=1,perplexity=30):
        """
        Project a 4D array onto a 1D array using specified manifold learning method.
        
        Parameters:
        data (np.ndarray): Input array
        method (str): Method to use for projection. Options are:
                    'pca-tsne', 'umap'
        scale_factor (float): Scale factor to multiply the final projection.
        output_dim (int): Number of dimensions to project to. Returns the nth component
                        if output_dim=n.
        
        Returns:
        np.ndarray: Output array of shape (N,) after projection, containing only the nth component.
        """
        if method == 'pca_tsne':
            premodel = PCA(n_components=50)
            model = TSNE(n_components=output_dim, method="exact",perplexity=perplexity)
        elif method == 'umap':
            from umap import UMAP
            premodel = PCA(n_components=50)
            model = UMAP(n_components=output_dim)
        else:
            raise ValueError("Invalid method specified. Choose from 'isomap', 'lle', 'hessian_lle', 'laplacian_eigenmaps', 'tsne', 'umap', 'pca', 'lda'.")
        
        # Fit the model and transform the data
        data = premodel.fit_transform(data)
        transformed_data = model.fit_transform(data)

        # Normalize data
        transformed_data -= np.min(transformed_data, axis=0)
        transformed_data /= np.max(transformed_data, axis=0)
        if method == 'umap':
            transformed_data -= np.max(transformed_data, axis=0)
            transformed_data *= -1

        # Select the nth component and scale
        nth_component = transformed_data[:, output_dim - 1]  # Adjust for 0-based indexing
        return nth_component * scale_factor
