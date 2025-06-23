
from tools.debug import Debug
try:
    import cupy as cp
    import numpy as np
    # Check if at least one GPU is available
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_GPU = True
    else:
        raise RuntimeError("No GPU detected.")
except (ImportError, RuntimeError):
    # Fallback to numpy if cupy is not installed or no GPU is available
    import numpy as cp
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


    def compute_centroids(self, label_image,labels=None):
        """
        Computes the MNI space centroids for all regions in label_indices.
        :param image_labels: Numpy array of the label image.
        :return: Numpy array of MNI space centroids.
        """
        if labels is None:
            labels = np.unique(label_image)
        
        # Use center_of_mass with `index` argument to calculate all centroids at once
        centroids = np.array(center_of_mass(label_image, labels=label_image, index=labels))
        
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
        # Convert the input arrays to CuPy arrays
        parcellation_data_cp  = cp.asarray(parcellation_data_np)
        nodal_strength_map_cp = cp.zeros(parcellation_data_cp.shape)
        
        # Create a dictionary to map label indices to nodal similarity values
        label_to_similarity = {label: similarity for label, similarity in zip(label_indices, feature_arr)}
        
        # Use CuPy to fill the nodal_strength_map_cp using the label-to-similarity mapping
        for label, similarity in label_to_similarity.items():
            nodal_strength_map_cp[parcellation_data_cp == label] = similarity
        
        # Set values where parcellation_data_cp == 0 to 0
        nodal_strength_map_cp[parcellation_data_cp == 0] = 0
        
        # Convert the result back to a NumPy array if needed
        nodal_strength_map_np = cp.asnumpy(nodal_strength_map_cp)
        return nodal_strength_map_np



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
            model = TSNE(n_components=output_dim, method="exact", n_iter=5000,perplexity=perplexity)
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