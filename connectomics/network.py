import numpy as np
from tools.debug import Debug
import networkx as nx
from multiprocessing import Pool, cpu_count
import bct

debug  = Debug()

class NetBasedAnalysis:
    def __init__(self) -> None:
        pass


    def binarize(self, simmatrix, threshold, mode="abs", threshold_mode="value", binarize=True):
        binarized = np.zeros(simmatrix.shape)
        
        if threshold_mode == "density":
            threshold = self.threshold_density(simmatrix, threshold)
        elif threshold_mode == "value":
            pass
        if mode == "posneg":
            if binarize:
                binarized[np.abs(simmatrix) < threshold] = 0
                binarized[np.abs(simmatrix) >= threshold] = np.sign(simmatrix[np.abs(simmatrix) >= threshold])
            else:
                binarized[np.abs(simmatrix) < threshold] = 0
                binarized[np.abs(simmatrix) >= threshold] = simmatrix[np.abs(simmatrix) >= threshold]
        elif mode == "abs":
            if binarize:
                binarized[np.abs(simmatrix) <= threshold] = 0
                binarized[np.abs(simmatrix) > threshold] = 1
            else:
                binarized[np.abs(simmatrix) <= threshold] = 0
                binarized[np.abs(simmatrix) > threshold] = simmatrix[np.abs(simmatrix) > threshold]
        elif mode == "pos":
            binarized[simmatrix < threshold] = 0
            if threshold_mode == "density":
                threshold = self.threshold_density(simmatrix, threshold)
            if binarize:
                binarized[simmatrix >= threshold] = 1
            else:
                binarized[simmatrix >= threshold] = simmatrix[simmatrix >= threshold]
        elif mode == "neg":
            binarized[simmatrix > threshold] = 0
            if threshold_mode == "density":
                threshold = self.threshold_density(simmatrix, threshold)
            if binarize:
                binarized[simmatrix <= threshold] = 1
            else:
                binarized[simmatrix <= threshold] = simmatrix[simmatrix <= threshold]
        # debug.info(threshold)
        return binarized

    def threshold_density(self, matrix, density):
        """
        Computes the threshold value based on the top X% density weights.

        Parameters:
            matrix (np.ndarray): The input similarity matrix.
            density (float): The percentage density for thresholding (e.g., 0.02 for 2%).

        Returns:
            float: The computed threshold value.
        """
        if density < 0 or density > 1:
            raise ValueError("Density must be a value between 0 and 1.")
        
        # Flatten the matrix to sort and find the threshold value
        flattened = matrix.flatten()
        
        # Number of elements to include based on density
        num_elements = int(np.ceil(density * len(flattened)))
        
        # Find the threshold value
        threshold_value = np.partition(flattened, -num_elements)[-num_elements]
        
        return threshold_value

    def random_graph_richclub(self, args):
        """
        Helper function to generate one randomized graph (via degree-preserving double-edge swap)
        and compute its rich-club coefficients at the specified degree thresholds using
        NetworkX's built-in function.
        
        Parameters
        ----------
        args : tuple
            A tuple containing:
                - G_obs (networkx.Graph): The observed graph.
                - degrees (numpy.ndarray): Array of degree thresholds.
                - nswap (int): Number of edge swaps to perform.
                - max_tries (int): Maximum number of attempts for the edge swap.
        
        Returns
        -------
        list
            A list of rich-club coefficients corresponding to each degree threshold.
        """
        G_obs, degrees, nswap = args
        # Create a randomized copy via degree-preserving double-edge swap.
        # G_rand = G_obs.copy()
        W = nx.to_numpy_array(G_obs)
        W_rand,_ = bct.randmio_und(W,nswap)
        G_rand = nx.to_networkx_graph(W_rand)
        # try:
        #     nx.double_edge_swap(G_rand, nswap=nswap, max_tries=max_tries)
        # except nx.NetworkXError:
        #     # If the swap procedure fails, use the current state of G_rand.
        #     pass

        # Compute the rich club coefficients using the built-in function.
        rc_dict = nx.rich_club_coefficient(G_rand, normalized=False)
        
        # For each specified threshold, ensure that at least two nodes have degree >= k;
        # otherwise, assign np.nan.
        rc_rand = []
        degree_dict_rand = dict(G_rand.degree())
        for k in degrees:
            if sum(1 for d in degree_dict_rand.values() if d >= k) < 2:
                rc_rand.append(np.nan)
            else:
                # Use the computed value if available; if not, default to np.nan.
                rc_rand.append(rc_dict.get(k, np.nan))
        return rc_rand

    def compute_richclub_stats(self, adj_matrix, num_random=100,alpha = 0.05,nswap=20):
        """
        Computes the rich-club coefficient of an observed graph and compares it 
        against a set of degree-preserving randomized graphs, using parallelization.

        Parameters
        ----------
        adj_matrix : numpy.ndarray
            A binarized (0/1) adjacency matrix representing an undirected graph.
        num_random : int, optional
            The number of randomized graphs to generate (default is 1000).

        Returns
        -------
        degrees : numpy.ndarray
            Array of degree thresholds (k) at which the rich club coefficients 
            were computed (only thresholds where at least two nodes have degree >= k).
        rc_coefficients : numpy.ndarray
            Observed rich-club coefficients computed for each degree threshold.
        mean_random_rc : numpy.ndarray
            Mean rich-club coefficient over the randomized graphs at each threshold.
        std_random_rc : numpy.ndarray
            Standard deviation of the rich-club coefficient over the randomized 
            graphs at each threshold.
        p_values : numpy.ndarray
            P-values (one-tailed) at each degree threshold, testing whether the observed
            rich-club coefficient is significantly greater than the randomized ones.
        """
        # Create the observed graph from the adjacency matrix.
        G_obs = nx.from_numpy_array(adj_matrix)
        self_loops = list(nx.selfloop_edges(G_obs))
        G_obs.remove_edges_from(self_loops)

        # Compute the observed rich club coefficients using the built-in function.
        rc_obs_dict = nx.rich_club_coefficient(G_obs, normalized=False)
        # Retrieve the node degrees and determine the valid thresholds (k) where
        # at least two nodes have degree >= k.
        degree_dict = dict(G_obs.degree())
        valid_thresholds = []
        obs_rc = []
        for k in sorted(rc_obs_dict.keys()):
            if sum(1 for d in degree_dict.values() if d >= k) < 2:
                continue  # Skip thresholds where the rich club is not defined.
            valid_thresholds.append(k)
            obs_rc.append(rc_obs_dict[k])
        degrees = np.array(valid_thresholds)
        rc_coefficients = np.array(obs_rc)
        # Set parameters for the double-edge swap (heuristic: 10 swaps per edge).
        # num_edges_obs = G_obs.number_of_edges()/G_obs.number_of_nodes()
        # nswap         = round(10 * num_edges_obs)
        # max_tries = nswap * 10
        # Prepare a list of arguments for each randomized graph.
        args_list = [(G_obs, degrees, nswap) for _ in range(num_random)]
        # Parallelize the computation over randomized graphs.
        with Pool(processes=cpu_count()) as pool:
            # Each call returns a list of rich-club coefficients for each threshold.
            rand_rc_list = pool.map(self.random_graph_richclub, args_list)
        # Convert the list to a NumPy array with shape (len(degrees), num_random).
        rand_rc_all      = np.array(rand_rc_list).T
        median_random_rc = np.array([np.nanmedian(np.unique(row)) for row in rand_rc_all])
        lower_bound      = np.array([np.nanpercentile(np.unique(row), (alpha/2)*100) for row in rand_rc_all])
        upper_bound      = np.array([np.nanpercentile(np.unique(row), (1 - alpha/2)*100) for row in rand_rc_all])
        # Compute p-values: fraction of randomized graphs where φ_rand >= φ_obs.
        p_values = np.zeros(len(degrees))
        for idx in range(len(degrees)):
            valid_samples = ~np.isnan(rand_rc_all[idx, :])
            if np.sum(valid_samples) > 0:
                count_ge = np.sum(rand_rc_all[idx, valid_samples] >= rc_coefficients[idx])
                p_values[idx] = (count_ge + 1) / (np.sum(valid_samples) + 1)  # small correction to avoid 0 p-values
            else:
                p_values[idx] = np.nan

        rand_rc_params = {"null_dist":rand_rc_all,
                          "median":median_random_rc,
                          "lower":lower_bound,
                          "upper":upper_bound,
                          "pvalue":p_values}
        return degrees, rc_coefficients, rand_rc_params










