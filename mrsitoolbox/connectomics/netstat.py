
import numpy as np
import cupy as cp
from os.path import split, join, exists
import nibabel as nib
from ..tools.filetools import FileTools
from ..tools.datautils import DataUtils
from ..tools.debug import Debug
from scipy.stats import pearsonr, spearmanr
import warnings
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
from scipy.stats import pearsonr
from scipy.optimize import minimize


debug    = Debug()



class NetStat:
    def __init__(self):
        pass 

    def generate_permuted_labels_by_adjacency(self,labels, adjacency_dict, n_swaps=None):
        """
        Generate a mapping from original labels to permuted labels by swapping labels among adjacent labels.

        Parameters
        ----------
        labels : list or array
            List of labels.
        adjacency_dict : dict
            Dictionary where keys are labels and values are sets of adjacent labels.
        n_swaps : int, optional
            Number of swaps to perform. If None, defaults to len(labels) * 10.

        Returns
        -------
        permuted_label_mapping : dict
            Dictionary mapping original labels to permuted labels.
        """
        if n_swaps is None:
            n_swaps = len(labels) * 10  # Arbitrary number of swaps

        label_mapping = {label: label for label in labels}

        for _ in range(n_swaps):
            # Randomly select a label
            label = np.random.choice(labels)
            # Get its adjacent labels
            adjacent_labels = list(adjacency_dict[label])
            if not adjacent_labels:
                continue  # No adjacent labels to swap with
            # Randomly select one of its adjacent labels
            swap_label = np.random.choice(adjacent_labels)
            # Swap the mapping of the two labels
            label_mapping[label], label_mapping[swap_label] = label_mapping[swap_label], label_mapping[label]

        return label_mapping

    def spatial_permutation_test(self,
        matrix_A,
        matrix_B,
        parcel_xyz_coords,
        label_indices,
        adjacency_dict,
        n_permutations=1000,
        random_state=None,
        corr_type="spearman",
    ):
        """
        Performs an adjacency-constrained permutation test to assess the statistical significance
        of the relationship between spatial distances and correlations
        (positive or negative) in brain imaging data, ensuring unique permutations.

        [Docstring remains the same as before]

        """
        if corr_type=="spearman":
            corr_stat = spearmanr
        elif corr_type=="pearson":
            corr_stat = pearsonr
        if random_state is not None:
            np.random.seed(random_state)
        n_nodes = parcel_xyz_coords.shape[0]
        # Map labels to indices and indices to labels
        label_to_index = {label_indices[i]: i for i in range(n_nodes)}
        index_to_label = {i: label_indices[i] for i in range(n_nodes)}
        labels = label_indices
        # Compute the upper triangle indices (excluding the diagonal)
        triu_indices = np.triu_indices(n_nodes, k=1)
        # Extract the upper triangle of the correlation matrix and distance matrix
        matrix_A_tri      = matrix_A[triu_indices]
        matrix_B_tri      = matrix_B[triu_indices]
        # Fit the distance model for the specified correlations and get goodness-of-fit
        observed_corr = corr_stat(matrix_A_tri.flatten(),matrix_B_tri.flatten()).statistic
        # Initialize array to store permutation results
        perm_corrs     = np.zeros(n_permutations)
        # Initialize a set to store unique permutations
        unique_permutations = set()
        # Permutation loop with progress bar
        perm = 0
        max_attempts = n_permutations * 10  # Maximum attempts to find unique permutations
        attempts = 0

        with tqdm(total=n_permutations, desc='Permutations') as pbar:
            while perm < n_permutations and attempts < max_attempts:
                # Generate a permuted label mapping by swapping labels among adjacent parcels
                permuted_label_mapping = self.generate_permuted_labels_by_adjacency(
                    labels, adjacency_dict, n_swaps=n_nodes*10
                )
                # Compute permuted indices
                permuted_indices = [label_to_index[permuted_label_mapping[index_to_label[i]]] for i in range(n_nodes)]
                # Convert permuted_indices to a tuple to make it hashable
                permuted_indices_tuple = tuple(permuted_indices)
                # Check if this permutation is unique
                if permuted_indices_tuple in unique_permutations:
                    attempts += 1
                    continue  # Skip and try again
                unique_permutations.add(permuted_indices_tuple)
                try:
                    # Permute the similarity matrix accordingly
                    permuted_simmatrix = matrix_A[np.ix_(permuted_indices, permuted_indices)]
                    # Extract the upper triangle of the permuted matrix
                    permuted_matrix = permuted_simmatrix[triu_indices]
                    perm_ids        = np.arange(permuted_matrix.size)
                    perm_matrix_A   = permuted_matrix[perm_ids]
                    perm_matrix_B   = matrix_B_tri[perm_ids]
                    # Fit the distance model for permuted correlations
                    perm_corrs[perm] = corr_stat(perm_matrix_A.flatten(),perm_matrix_B.flatten()).statistic
                    perm += 1
                    attempts = 0  # Reset attempts after a successful permutation
                    pbar.update(1)
                except Exception as e:
                    # Optionally, handle exceptions or log errors
                    attempts += 1
                    pass

        if perm < n_permutations:
            debug.warning(f"Only {perm} unique permutations were generated after {max_attempts} attempts.")

        # Compute p-value for the specified correlations
        p_value = np.mean(perm_corrs[:perm] >= observed_corr)
        return observed_corr,p_value

    @staticmethod
    def fit_exponential_model(M, S):
        """
        Fits the model M = exp(g * S) by finding the optimal g.
        
        Parameters:
        M (numpy.ndarray): Observed matrix.
        S (numpy.ndarray): Input matrix for the model.
        
        Returns:
        tuple: (g_optimal, corr), where g_optimal is the best-fit parameter and 
            corr is the correlation coefficient between observed M and fitted M.
        """
        # Define the loss function: Mean Squared Error
        def mse_loss(g):
            fitted_M = np.exp(g * S)
            return np.mean((M - fitted_M) ** 2)
        # Optimize g to minimize the MSE
        result = minimize(mse_loss, x0=0, method='BFGS')  # x0=0 is the initial guess for g
        g_optimal = result.x[0]
        # Compute the fitted matrix with the optimal g
        fitted_M = np.exp(g_optimal * S)
        # Compute the correlation coefficient between observed M and fitted M
        corr = np.corrcoef(M.flatten(), fitted_M.flatten())[0, 1]
        return g_optimal, corr
