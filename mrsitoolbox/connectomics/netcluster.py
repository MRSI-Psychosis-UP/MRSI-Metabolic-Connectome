import numpy as np
import copy
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from scipy.spatial.distance import squareform
import community as community_louvain
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, AffinityPropagation
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import linregress, spearmanr
from .parcellate import Parcellate
from ..tools.debug import Debug
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
from tqdm import tqdm
import warnings
# Suppress the specific warning from sklearn
warnings.filterwarnings(
    "ignore",
    message="Graph is not fully connected, spectral embedding may not work as expected.",
    category=UserWarning,
    module="sklearn.manifold._spectral_embedding"
)


import networkx.algorithms.community as nx_comm
METABOLITES     = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
FONTSIZE        = 16

parc      = Parcellate()
debug     = Debug()

class NetCluster(object):
    def __init__(self):
        self.base_labels = None
        pass

    def consensus_clustering(self,similarity_matrix, n_clusters, n_iterations=10):
        """
        Perform consensus clustering on a similarity matrix using multiple clustering algorithms.
        
        Parameters:
        - similarity_matrix: 2D numpy array representing the similarity matrix.
        - n_clusters: Number of clusters.
        - n_iterations: Number of clustering runs to perform for each algorithm.
        
        Returns:
        - consensus_labels: Consensus cluster labels for each data point.
        """
        n_samples = similarity_matrix.shape[0]
        all_labels = np.zeros((n_iterations * 4, n_samples), dtype=int)

        # Convert similarity matrix to distance matrix
        distance_matrix = 1 - similarity_matrix

        clustering_algorithms = [
            KMeans(n_clusters=n_clusters, n_init=1),
            AgglomerativeClustering(n_clusters=n_clusters),
            SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize')
        ]

        idx = 0
        for alg in clustering_algorithms:
            for _ in range(n_iterations):
                if isinstance(alg, SpectralClustering):
                    labels = alg.fit_predict(similarity_matrix)
                else:
                    labels = alg.fit_predict(distance_matrix)
                all_labels[idx, :] = labels
                idx += 1

        # Add Louvain algorithm results
        for _ in range(n_iterations):
            # Convert similarity matrix to a weighted graph
            graph = nx.from_numpy_array(similarity_matrix)
            partition = community_louvain.best_partition(graph)
            labels = np.array([partition[i] for i in range(n_samples)])
            all_labels[idx, :] = labels
            idx += 1

        # Build the co-association matrix
        co_association_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            for j in range(n_samples):
                co_association_matrix[i, j] = np.sum(all_labels[:, i] == all_labels[:, j])

        co_association_matrix /= (n_iterations * (len(clustering_algorithms) + 1))

        # Perform hierarchical clustering on the co-association matrix
        condensed_dist_matrix = squareform(1 - co_association_matrix)
        linkage_matrix = sch.linkage(condensed_dist_matrix, method='average')
        consensus_labels = sch.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

        return consensus_labels

    def normalize_2Dconnectivity(self,weighted_simmatrix):
        norms = np.linalg.norm(weighted_simmatrix, axis=2, keepdims=True)
        return weighted_simmatrix / norms


    def get_structural_simmilarity(self,weighted_simmatrix2D):
        features = np.hstack((weighted_simmatrix2D[:,:,0], weighted_simmatrix2D[:,:,1]))
        n_nodes = weighted_simmatrix2D.shape[0]
        structural_simmilarity = np.zeros(weighted_simmatrix2D[:,:,0].shape)
        for i in tqdm(range(n_nodes)):
            A      = features[i,:]
            A_norm = np.linalg.norm(A)
            for j in range(n_nodes):
                if i>j:
                    B           =  features[j,:]
                    B_norm      = np.linalg.norm(B)
                    struct_sim  = np.dot(A,B)/(A_norm*B_norm)
                    structural_simmilarity[i,j] = struct_sim
                    structural_simmilarity[j,i] = struct_sim
        return structural_simmilarity


    def __get_structural_simmilarity(self,weighted_simmatrix):
        # Reshape the normalized correlations for efficient computation
        n = weighted_simmatrix.shape[0]
        normalized_correlations_reshaped = weighted_simmatrix.reshape(n, -1)
        # Compute cosine similarity matrix using matrix multiplication
        cos_sim_matrix = normalized_correlations_reshaped @ normalized_correlations_reshaped.T
        # Ensure diagonal is 1 (self-similarity)
        np.fill_diagonal(cos_sim_matrix, 1)
        return cos_sim_matrix

    def transform_to_4d(self,weighted_simmatrix):
        n = weighted_simmatrix.shape[0]
        transformed_matrix = np.zeros((n, n, 4))

        sim1 = weighted_simmatrix[:,:,0]
        sim2 = weighted_simmatrix[:,:,1]

        transformed_matrix[sim1 > 0, 0] =  sim1[sim1 > 0]
        transformed_matrix[sim1 < 0, 1] = -sim1[sim1 < 0]
        transformed_matrix[sim2 > 0, 2] =  sim2[sim2 > 0]
        transformed_matrix[sim2 < 0, 3] = -sim2[sim2 < 0]

        return transformed_matrix

    def modularize(self,matrix,permuted_ids=None):
        if permuted_ids is None:
            permuted_ids = reverse_cuthill_mckee(csr_matrix(matrix))
        permuted_matrix = copy.deepcopy(matrix)
        permuted_matrix = permuted_matrix[permuted_ids,:]
        permuted_matrix = permuted_matrix[:,permuted_ids]
        return permuted_matrix, permuted_ids


    def louvain_clustering(self, weighted_simmatrix, resolution=1, mode="equivalence"):          
        G = nx.Graph()
        n = weighted_simmatrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if weighted_simmatrix[i, j] > 0:  # You may define a threshold here if needed
                    G.add_edge(i, j, weight=weighted_simmatrix[i, j])
        
        partition = nx_comm.louvain_partitions(G, resolution=resolution)
        # Get the final partition
        final_partition = list(partition)[-1]
        labels = np.zeros(n, dtype=int)
        for idx, community in enumerate(final_partition):
            for node in community:
                labels[node] = idx

        if self.base_labels is None:
            self.base_labels = labels.copy()
        else:
            labels = self.relabel_clusters(labels, self.base_labels)
        
        return labels



    def get_homotopy(self,data,scale_factor=255.0,output_dim=1,perplexity=30):
        premodel = PCA(n_components=50)
        model    = TSNE(n_components=output_dim, method="exact", 
                        n_iter=5000,perplexity=perplexity)
        # Fit the model and transform the data
        data = premodel.fit_transform(data)
        transformed_data = model.fit_transform(data)
        kl_div = model.kl_divergence_
        # Normalize data
        transformed_data -= np.min(transformed_data, axis=0)
        transformed_data /= np.max(transformed_data, axis=0)
        # Select the nth component and scale
        nth_component = transformed_data[:, output_dim - 1]  # Adjust for 0-based indexing
        return nth_component * scale_factor, kl_div

    def project_4d_to_1d(self, data, method='isomap', scale_factor=255.0, output_dim=1,perplexity=30):
        """
        Project a 4D array onto a 1D array using specified manifold learning method.
        
        Parameters:
        data (np.ndarray): Input array of shape (N, 4).
        method (str): Method to use for projection. Options are:
                    'isomap', 'lle', 'hessian_lle', 'laplacian_eigenmaps',
                    'tsne', 'umap', 'pca', 'lda'
        scale_factor (float): Scale factor to multiply the final projection.
        output_dim (int): Number of dimensions to project to. Returns the nth component
                        if output_dim=n.
        
        Returns:
        np.ndarray: Output array of shape (N,) after projection, containing only the nth component.
        """
        if method == 'isomap':
            model = Isomap(n_components=output_dim)
        elif method == 'lle':
            model = LocallyLinearEmbedding(n_components=output_dim)
        elif method == 'hessian_lle':
            model = LocallyLinearEmbedding(n_components=output_dim, method='hessian')
        elif method == 'laplacian_eigenmaps':
            from sklearn.manifold import SpectralEmbedding
            model = SpectralEmbedding(n_components=output_dim)
        elif method == 'tsne':
            model = TSNE(n_components=output_dim, method="exact", n_iter=5000)
        elif method == 'pca_tsne':
            premodel = PCA(n_components=50)
            model = TSNE(n_components=output_dim, method="exact", n_iter=5000,perplexity=perplexity)
        elif method == 'umap':
            from umap import UMAP
            premodel = PCA(n_components=50)
            model = UMAP(n_components=output_dim)
        elif method == 'pca':
            model = PCA(n_components=output_dim)
        elif method == 'lda':
            model = LDA(n_components=output_dim)
        else:
            raise ValueError("Invalid method specified. Choose from 'isomap', 'lle', 'hessian_lle', 'laplacian_eigenmaps', 'tsne', 'umap', 'pca', 'lda'.")
        
        # Fit the model and transform the data
        if method == 'pca_tsne' or method == 'umap':
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

    

    def cluster_all_algorithms(self,X,n_clusters=3, random_state=42):
        """
        Clusters the given Nx4 dataset using all available clustering algorithms in scikit-learn.
        
        Parameters:
            X (numpy.ndarray or pandas.DataFrame): The Nx4 dataset to cluster.
            
        Returns:
            dict: A dictionary where keys are the names of clustering algorithms and values are the cluster labels.
        """
      
        # Initialize clustering algorithms
        clustering_algorithms = {
            # 'KMeans': KMeans(n_clusters=n_clusters, random_state=random_state),
            # 'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
            'SpectralClustering': SpectralClustering(n_clusters=n_clusters, 
                                                     random_state=random_state,n_init=500,
                                                     affinity="nearest_neighbors"),
            # 'Birch': Birch(n_clusters=n_clusters),
            'GaussianMixture': GaussianMixture(n_components=n_clusters, random_state=random_state),
            # 'AffinityPropagation': AffinityPropagation(random_state=random_state)
        }
        
        # Dictionary to store the cluster labels
        cluster_labels = {}
        
        # Fit each algorithm and get the cluster labels
        for name, algorithm in clustering_algorithms.items():
            try:
                if name == 'AgglomerativeClustering':
                    algorithm.fit(X)
                    labels = algorithm.fit_predict(X)
                else:
                    labels = algorithm.fit_predict(X)
                cluster_labels[name] = labels
            except Exception as e:
                cluster_labels[name] = f"Error: {e}"
    
        # cluster_labels["majority"] = self.majority_labels(cluster_labels)
        # cluster_labels["consensus"] = self.consensus_clustering(cluster_labels, n_clusters)
        try:
            cluster_labels["monti"]     = self.monti_consensus_clustering(cluster_labels, n_clusters)
        except Exception as e:
            debug.warning("Monti consensus failed",e)
        return cluster_labels

    def gaussian_mixture(self,X,n_clusters=3, random_state=42):
        clust_alg = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels    = clust_alg.fit_predict(X)
        return labels


    def relabel_clusters(self,new_labels, base_labels):
        """
        Relabels the target_labels to match the reference_labels as closely as possible.
        
        Parameters:
            reference_labels (numpy.ndarray): The reference cluster labels.
            target_labels (numpy.ndarray): The target cluster labels to relabel.
            
        Returns:
            numpy.ndarray: The relabeled target cluster labels.
        """
        new_labels_unique = np.unique(new_labels)
        base_labels_unique = np.unique(base_labels)
        
        cost_matrix = np.zeros((len(new_labels_unique), len(base_labels_unique)))

        for i, new_label in enumerate(new_labels_unique):
            for j, base_label in enumerate(base_labels_unique):
                cost_matrix[i, j] = np.sum((new_labels == new_label) & (base_labels == base_label))

        row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
        
        new_labels_aligned = new_labels.copy()
        label_mapping = dict(zip(new_labels_unique[row_ind], base_labels_unique[col_ind]))

        for new_label, base_label in label_mapping.items():
            new_labels_aligned[new_labels == new_label] = base_label

        return new_labels_aligned

    def majority_labels(self,cluster_results):
        """
        Determines the majority label for each data point across multiple clustering algorithms.
        
        Parameters:
            cluster_results (dict): A dictionary where keys are the names of clustering algorithms and
                                    values are the cluster labels.
                                    
        Returns:
            numpy.ndarray: An array of majority labels for each data point.
        """
        # Get the number of data points
        num_points = len(next(iter(cluster_results.values())))
        
        # Initialize a list to store majority labels
        majority_labels = []
        
        for i in range(num_points):
            # Collect labels for the ith data point from each algorithm
            labels = [labels[i] for labels in cluster_results.values() if not isinstance(labels, str)]
            
            # Determine the majority label
            most_common_label, _ = Counter(labels).most_common(1)[0]
            
            majority_labels.append(most_common_label)
        
        return np.array(majority_labels)


    def monti_consensus_clustering(self, cluster_labels, n_clusters):
        """
        Determine the consensus clustering using the Monti Consensur algorithm.
        
        Parameters:
            cluster_labels (dict): A dictionary where keys are algorithm names and values are the cluster labels.
            n_clusters (int): The number of clusters to find in the consensus.
            
        Returns:
            np.ndarray: The consensus clustering labels.
        """
        # Extract cluster labels from the dictionary
        clusterings = [labels for labels in cluster_labels.values() if isinstance(labels, np.ndarray)]
        n_samples = clusterings[0].shape[0]
        n_clusterings = len(clusterings)
        
        # Step 1: Create co-association matrix
        co_association_matrix = np.zeros((n_samples, n_samples))
        
        for clustering in clusterings:
            for i in range(n_samples):
                for j in range(n_samples):
                    if clustering[i] == clustering[j]:
                        co_association_matrix[i, j] += 1
        
        co_association_matrix /= n_clusterings
        
        # Step 2: Apply spectral clustering on the co-association matrix
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        consensus_labels = spectral.fit_predict(co_association_matrix)
        
        return consensus_labels        

    def cluster_correlations(self,parcel_concentrations3D,cluster_x_id,cluster_y_id):
        x_array = parcel_concentrations3D[cluster_x_id].flatten()
        y_array = parcel_concentrations3D[cluster_y_id].flatten()
        res = spearmanr(x_array, y_array)
        res2 = parc.speanman_corr_quadratic(x_array, y_array)
        corr1 = res.statistic
        corr2 = round(res2["corr"],2)
        return corr1,corr2
    

    def optimal_gmm_clusters(self,X, max_clusters=10):
        """
        Determines the optimal number of clusters for a Gaussian Mixture Model using BIC and AIC.
        
        Parameters:
            X (numpy.ndarray or pandas.DataFrame): The dataset to cluster.
            max_clusters (int): The maximum number of clusters to consider.
            
        Returns:
            dict: A dictionary with keys 'n_clusters', 'bic', 'aic', and 'models', containing the optimal number of clusters,
                the BIC and AIC values for each number of clusters, and the fitted GMM models.
        """
       
        # Initialize lists to store BIC and AIC values
        bics = []
        aics = []
        models = []
        
        # Fit GMM for different numbers of clusters
        for n_clusters in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(X)
            bics.append(gmm.bic(X))
            aics.append(gmm.aic(X))
            models.append(gmm)
        
        # Determine the optimal number of clusters
        optimal_n_clusters_bic = np.argmin(bics) + 1
        optimal_n_clusters_aic = np.argmin(aics) + 1
        
        # Plot BIC and AIC values
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, max_clusters + 1), bics, label='BIC', marker='o')
        plt.plot(range(1, max_clusters + 1), aics, label='AIC', marker='o')
        plt.axvline(optimal_n_clusters_bic, color='r', linestyle='--', label=f'Optimal BIC: {optimal_n_clusters_bic}')
        plt.axvline(optimal_n_clusters_aic, color='g', linestyle='--', label=f'Optimal AIC: {optimal_n_clusters_aic}')
        plt.xlabel('Number of clusters')
        plt.ylabel('Criterion Value')
        plt.title('BIC and AIC for Gaussian Mixture Model')
        plt.legend()
        plt.show()
        
        return {
            'n_clusters_bic': optimal_n_clusters_bic,
            'n_clusters_aic': optimal_n_clusters_aic,
            'bic': bics,
            'aic': aics,
            'models': models
        }

# Example usage
if __name__ == "__main__":
    netclust = NetCluster()
    # Example similarity matrix (symmetric with 1s on the diagonal)
    similarity_matrix = np.array([
        [1.0, 0.8, 0.2, 0.3],
        [0.8, 1.0, 0.4, 0.5],
        [0.2, 0.4, 1.0, 0.6],
        [0.3, 0.5, 0.6, 1.0]
    ])

    n_clusters = 2
    consensus_labels = netclust.consensus_clustering(similarity_matrix, n_clusters)
    print("Consensus Cluster Labels:", consensus_labels)
