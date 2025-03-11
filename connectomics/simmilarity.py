try:
    import cupy as cp
    # Check if at least one GPU is available
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_GPU = True
    else:
        raise RuntimeError("No GPU detected.")
except (ImportError, RuntimeError):
    # Fallback to numpy if cupy is not installed or no GPU is available
    import numpy as cp
    USE_GPU = False
    
import numpy as np
from tools.debug import Debug
import os , math, copy


debug    = Debug()


class Simmilarity:
    def __init__(self):
        pass

    def filter_sparse_matrices(self,matrix_list):
        n_zeros_arr = list()
        for i,sim in enumerate(matrix_list):
            n_zeros = len(np.where(sim==0)[0])
            n_zeros_arr.append(n_zeros)

        n_zeros_arr = np.array(n_zeros_arr)
        debug.info("0 nodal strength count",n_zeros_arr.mean(),"+-",n_zeros_arr.std())

        include_indices = list()
        exclude_indices = list()
        matrix_list_refined = list()

        for i,sim in enumerate(matrix_list):
            n_zeros = len(np.where(sim==0)[0])
            if n_zeros<n_zeros_arr.mean()+n_zeros_arr.std():
                matrix_list_refined.append(sim)
                include_indices.append(i)
            else:
                exclude_indices.append(i)
        return matrix_list_refined,include_indices,exclude_indices


    @staticmethod
    def nodal_strength_map(nodal_similarity_matrix, parcellation_data_np, label_indices):
        # Convert the input arrays to CuPy arrays
        parcellation_data_cp  = cp.asarray(parcellation_data_np)
        nodal_strength_map_cp = cp.zeros(parcellation_data_cp.shape)
        
        # Create a dictionary to map label indices to nodal similarity values
        label_to_similarity = {label: similarity for label, similarity in zip(label_indices, nodal_similarity_matrix)}
        
        # Use CuPy to fill the nodal_strength_map_cp using the label-to-similarity mapping
        for label, similarity in label_to_similarity.items():
            nodal_strength_map_cp[parcellation_data_cp == label] = similarity
        
        # Set values where parcellation_data_cp == 0 to 0
        nodal_strength_map_cp[parcellation_data_cp == 0] = 0
        
        # Convert the result back to a NumPy array if needed
        nodal_strength_map_np = cp.asnumpy(nodal_strength_map_cp)
        return nodal_strength_map_np

    @staticmethod
    def __nodal_strength_map(nodal_similarity_matrix, parcellation_data_np, label_indices):
        """
        Compute the nodal strength map using GPU (with CuPy) if available, otherwise fallback to CPU (with NumPy).
        
        Args:
            nodal_similarity_matrix (array): Nodal similarity values.
            parcellation_data_np (array): Parcellation data array.
            label_indices (array): Array of label indices.
        
        Returns:
            np.ndarray: Nodal strength map array.
        """

        # Move data to the appropriate device
        parcellation_data_cp = cp.asarray(parcellation_data_np)
        similarity_cp = cp.asarray(nodal_similarity_matrix)
        labels_cp = cp.asarray(label_indices)

        # Determine max label to build a lookup table
        max_label = int(labels_cp.max())
        # Create lookup table
        lookup = cp.zeros((max_label + 1,), dtype=similarity_cp.dtype)
        lookup[labels_cp] = similarity_cp

        # Compute the nodal strength map
        nodal_strength_map_cp = lookup[parcellation_data_cp]

        # Ensure positions where parcellation_data_cp == 0 remain zero
        nodal_strength_map_cp[parcellation_data_cp == 0] = 0

        # Move back to CPU if using GPU
        nodal_strength_map_np = cp.asnumpy(nodal_strength_map_cp) if USE_GPU else nodal_strength_map_cp
        return nodal_strength_map_np


    def nodal_similarity(self,matrix):
        # Exclude the diagonal elements (self-connections) by setting them to zero
        np.fill_diagonal(matrix, 0)
        # Sum the weights of connections from each node to all other nodes
        similarities = np.sum(matrix, axis=1)
        return similarities

    def get_feature_nodal_similarity(self,simmatrix_matrix):
        simmatrix_pop_weighted_plus = copy.deepcopy(simmatrix_matrix)
        simmatrix_pop_weighted_neg  = copy.deepcopy(simmatrix_matrix)
        simmatrix_pop_weighted_plus[simmatrix_matrix<0] = 0
        simmatrix_pop_weighted_neg[simmatrix_matrix>0]  = 0
        NS_parcel_plus     = self.nodal_similarity(simmatrix_pop_weighted_plus)
        NS_parcel_neg      = self.nodal_similarity(simmatrix_pop_weighted_neg)
        # NS_map_plus        = self.nodal_strength_map(NS_parcel_plus,parcellation_data_np,label_indices)
        # NS_map_neg         = self.nodal_strength_map(NS_parcel_neg,parcellation_data_np,label_indices)
        # nodal_strength_map_np_scalar        = self.polar_to_angle(NS_map_plus, NS_map_neg)
        features     = np.zeros((NS_parcel_plus.shape)+(2,))
        features[:,0] = NS_parcel_plus
        features[:,1] = NS_parcel_neg
        return features

    def get_feature_similarity(self,simmatrix_matrix4D):
        features = np.hstack((simmatrix_matrix4D[:,:,0], simmatrix_matrix4D[:,:,1]))
        return features

    def get_4D_feature_nodal_similarity(self,weighted_metab_sim_4D_avg):
        features2D_1  = self.get_feature_nodal_similarity(weighted_metab_sim_4D_avg[:,:,0])
        features2D_2  = self.get_feature_nodal_similarity(weighted_metab_sim_4D_avg[:,:,1])
        features4D    = np.zeros((features2D_1.shape[0],)+(4,))
        features4D[:,0:2] = features2D_1
        features4D[:,2:4] = features2D_2
        return features4D


    def get_homotopy2(self,simmatrix_matrix,parcellation_data_np,label_indices):
        features_1 = self.get_homotopy(simmatrix_matrix[:,:,0],parcellation_data_np,label_indices)
        features_2 = self.get_homotopy(simmatrix_matrix[:,:,1],parcellation_data_np,label_indices)
        features_4d     = np.zeros((features_1.shape[0],)+(4,))
        features_4d[:,0] = features_1[:,0]
        features_4d[:,1] = features_1[:,1]
        features_4d[:,2] = features_2[:,0]
        features_4d[:,3] = features_2[:,1]
        return features_4d
    


