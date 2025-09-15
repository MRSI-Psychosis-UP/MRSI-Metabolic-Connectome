
from tools.debug import Debug
import networkx as nx
import itertools
import numpy as np
from multiprocessing import Pool
from rich.progress import Progress

debug = Debug()


from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

def _dfs_paths_from_neighbor(G_data, path, target, cutoff):
    G = nx.from_dict_of_lists(G_data)
    all_paths = []

    def dfs(current_path):
        current_node = current_path[-1]
        if current_node == target:
            all_paths.append(current_path)
            return
        if len(current_path) > cutoff:
            return
        for neighbor in G.neighbors(current_node):
            if neighbor not in current_path:  # avoid cycles
                dfs(current_path + [neighbor])

    dfs(path)
    return all_paths

class NetFibre:
    def __init__(self,homotopy_dict,centroid_dict,nodal_strength_dict,network_energy):
        self.homotopy_dict       = homotopy_dict
        self.centroid_dict       = centroid_dict
        self.network_energy      = network_energy
        self.nodal_strength_dict = nodal_strength_dict
        self.n_nodes             = len(centroid_dict)
        self.nbins_shannon       = round(2*self.n_nodes**(1/3))

    def cost_fn(self,path,beta=0.5):
        t              = np.linspace(0, 1, len(path))
        msi            = np.array([self.homotopy_dict[node] for node in path])
        gradient       = np.gradient(msi, t) * 1 / len(path)
        nodal_strength = np.array([self.nodal_strength_dict[node] for node in path])
        return np.nanmean(np.abs(gradient))-beta*np.mean(nodal_strength)/self.network_energy

    def find_all_simple_paths(self,graph_dict, source, target, cutoff=23):
        """
        Find all simple paths between two nodes in a graph.

        Parameters:
        - graph_dict (dict): A dictionary where keys are nodes and values are sets of adjacent nodes.
        - source: The starting node.
        - target: The target node.
        - directed (bool): Whether the graph should be treated as directed (default is False for undirected).

        Returns:
        - List[List]: A list of simple paths, each represented as a list of nodes.
        """
        G = self.construct_gm_graph(graph_dict)
        return list(nx.all_simple_paths(G, source=source, target=target,cutoff=cutoff))


    def find_all_simple_paths_parallel(self, graph_dict, source, target, cutoff=23, max_workers=28):
        """
        Find all simple paths between two nodes in a graph using parallel DFS with a progress bar.

        Parameters:
        - graph_dict (dict): A dictionary where keys are nodes and values are sets of adjacent nodes.
        - source: The starting node.
        - target: The target node.
        - cutoff (int): Maximum path length.
        - max_workers (int): Number of parallel workers.

        Returns:
        - List[List]: A list of simple paths, each represented as a list of nodes.
        """
        G = self.construct_gm_graph(graph_dict)
        G_data = nx.to_dict_of_lists(G)

        if source == target:
            return [[source]]
        if source not in G or target not in G:
            return []

        first_neighbors = list(G.neighbors(source))
        if not first_neighbors:
            return []

        tasks = [[source, nbr] for nbr in first_neighbors]
        all_paths = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:

            task_id = progress.add_task("Finding paths...", total=len(tasks))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_dfs_paths_from_neighbor, G_data, path, target, cutoff): path
                    for path in tasks
                }

                for future in as_completed(futures):
                    all_paths.extend(future.result())
                    progress.update(task_id, advance=1)

        return all_paths


    @staticmethod
    def find_simplest_path(adjacency_dict, start_label, end_label):
        """
        Finds the simplest path (shortest in terms of steps) between two nodes.

        Args:
            adjacency_dict (dict): A dictionary where keys are node labels (int),
                                and values are sets of adjacent nodes (int).
            start_label (int): Starting node label.
            end_label (int): Ending node label.

        Returns:
            List[int]: The simplest path as a list of node labels, or an empty list if no path exists.
        """
        from collections import deque
        # Queue for BFS: stores paths as lists
        queue = deque([[start_label]])
        visited = set()
        while queue:
            # Get the current path
            path = queue.popleft()
            current = path[-1]
            # If the current node is the target, return the path
            if current == end_label:
                return path
            # Mark the current node as visited
            visited.add(current)
            # Add neighbors to the queue if not visited
            for neighbor in sorted(adjacency_dict.get(current, [])):  # Sort for consistent order
                if neighbor not in visited and neighbor not in path:
                    queue.append(path + [neighbor])
        # Return empty list if no path found
        return []
    
    @staticmethod
    def construct_gm_graph(graph_dict):
        G = nx.Graph()
        # Add edges to the graph
        for node, neighbors in graph_dict.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G

    def interpolate_optimal_path(self, path_labels, adj_G, L=5, beta=0.5):
        """
        Constructs a complete path by finding and concatenating the best candidate paths 
        between consecutive labels in 'path_labels'. For each pair of nodes, the method 
        searches for all simple paths with exactly L nodes and selects the one that minimizes 
        the cost defined by self.cost_fn. Only candidate paths with at least two nodes that have 
        non-NaN homotopy values (according to self.homotopy_dict) are considered.
        
        If no candidate path is found between two nodes, the function attempts to use the 
        shortest path. If even that fails (i.e. the nodes are disconnected), the nodes are 
        added directly to the complete path.
        
        Args:
            path_labels (List[int]): A list of labels defining the points of interest.
            adj_G (networkx.Graph): The adjacency graph.
            L (int): Desired number of nodes for candidate paths.
            beta (float, optional): Parameter balancing the cost components.
        
        Returns:
            List[int]: The complete path as a list of node labels.
        """
        complete_path = []  # Initialize the complete path list
        for i in range(len(path_labels) - 1):
            start_label = path_labels[i]
            stop_label  = path_labels[i + 1]
            
            # Find all candidate simple paths with exactly L nodes and at least 2 valid homotopy values.
            candidate_paths = [
                p for p in nx.all_simple_paths(adj_G, start_label, stop_label, cutoff=L)
                if len(p) == L and all(not np.isnan(self.homotopy_dict[node]) for node in p)
            ]
            
            best_path = None
            if candidate_paths:
                # Compute the cost for each candidate path using self.cost_fn.
                costs = [self.cost_fn(p, beta) for p in candidate_paths]
                # Select the candidate with the minimal cost.
                best_path = candidate_paths[np.argmin(costs)]
            else:
                # Fallback: try to use the shortest path.
                try:
                    best_path = nx.shortest_path(adj_G, start_label, stop_label)
                except Exception:
                    best_path = None
            
            if best_path:
                if not complete_path:
                    # For the first segment, add the entire path.
                    complete_path.extend(best_path)
                else:
                    # For subsequent segments, avoid duplicating the starting node.
                    complete_path.extend(best_path[1:])
            else:
                # No path found between start_label and stop_label.
                if not complete_path:
                    complete_path.append(start_label)
                    complete_path.append(stop_label)
                else:
                    if complete_path[-1] != start_label:
                        complete_path.append(start_label)
                    complete_path.append(stop_label)
        complete_path = [node for node in complete_path if not np.isnan(self.homotopy_dict[node])]
        return complete_path



    def interpolate_complete_path(self, path_labels, adj_G):
        """
        Constructs a complete path by finding and concatenating the simplest paths 
        between consecutive labels in labels_points_neo_LH without duplicating nodes.
        If no path is found between two nodes, the disconnected nodes are added directly 
        to the complete path.

        Args:
            path_labels (List[int]): A list of labels defining the points of interest.
            adjacency_dict (dict): Adjacency dictionary for the network.

        Returns:
            List[int]: The complete path as a list of node labels.
        """
        complete_path = []  # Initialize the complete path list

        for i in range(len(path_labels) - 1):
            start_label = path_labels[i]
            stop_label = path_labels[i + 1]
            # Find the simplest path between consecutive nodes
            try:
                _path  = nx.shortest_path(adj_G, start_label, stop_label)
            except Exception: pass # slip if nodes are not connected
            # _path = self.find_simplest_path(adjacency_dict, start_label, stop_label)
            if _path:
                if not complete_path:
                    # For the first segment, add the entire path
                    complete_path.extend(_path)
                else:
                    # For subsequent segments, avoid duplicating the start node
                    complete_path.extend(_path[1:])
            else:
                # No path found between start_label and stop_label
                if not complete_path:
                    # If complete_path is empty, add both nodes
                    complete_path.append(start_label)
                    complete_path.append(stop_label)
                else:
                    # If the last node in complete_path isn't start_label, add start_label
                    if complete_path[-1] != start_label:
                        complete_path.append(start_label)
                    # Add stop_label to indicate disconnection
                    complete_path.append(stop_label)
        return complete_path

    def get_gradient_along_paths(self, all_paths):
        all_paths_list = list()
        for path in all_paths:
            homotopy_list = [self.homotopy_dict[node] for node in path]
            # Count the number of NaN values in homotopy_list
            nan_count = sum(1 for val in homotopy_list if np.isnan(val))
            # Only process the path if there are less than 2 NaN values
            if nan_count < 2:
                if self.centroid_dict is not None:
                    centroid_list = [self.centroid_dict[node] for node in path]
                else:
                    centroid_list = None
                all_paths_list.append((homotopy_list, centroid_list, path))
        return all_paths_list


    def has_path_between(self,graph_dict, source, target, directed=False):
        """
        Check if there is a path between two nodes in a graph using NetworkX.
        """
        # Create the appropriate graph type
        G = nx.Graph()
        # Add edges to the graph
        for node, neighbors in graph_dict.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        # Check if there's a path between source and target
        return nx.has_path(G, source, target)
    



    def perturb_path(self,graph_dict, path, npert=1):
        """
        Generate perturbations of a given simple path by replacing one internal node.
        
        For each internal node (i.e. not the first or last), the function looks for candidate
        replacement nodes that are adjacent to both the node's predecessor and successor in the path.
        The replacement is only accepted if:
        - It differs from the original node.
        - It does not already appear in the rest of the path (ensuring the path remains simple).
        
        Parameters:
        graph_dict (dict): A dictionary mapping each node to a set of its adjacent nodes.
        path (list): A list of nodes representing a simple path.
        npert (int): Maximum number of perturbations to generate.
        
        Returns:
        list: A list of perturbed paths (each path is a list of nodes).
        """
        perturbations = []
        # We only perturb internal nodes; endpoints are kept fixed.
        for i in range(1, len(path) - 1):
            prev_node = path[i - 1]
            next_node = path[i + 1]
            # Get candidate nodes that are adjacent to both prev_node and next_node.
            candidates = graph_dict.get(prev_node, set()).intersection(graph_dict.get(next_node, set()))
            for candidate in candidates:
                # Skip if candidate is the same as the current node.
                if candidate == path[i]:
                    continue
                # Ensure the candidate is not already in the path (except at the position being replaced).
                if candidate in path[:i] or candidate in path[i+1:]:
                    continue
                # Create a new perturbed path with the candidate replacing the original node.
                new_path = path.copy()
                new_path[i] = candidate
                perturbations.append(new_path)
                # Stop if we've reached the desired number of perturbations.
                if len(perturbations) >= npert:
                    return perturbations
        return perturbations

    def perturb_multiple_nodes_path(self,graph, path, npert=1):
        """
        Generate perturbations of a given simple path by simultaneously perturbing 
        one or more internal nodes.
        
        For each internal node (i.e. not the first or last node in the path),
        a candidate replacement is computed as the intersection of the neighbors 
        of its preceding and following nodes in the original path, excluding 
        the original node itself.
        
        Then, the function considers all combinations of internal nodes to perturb.
        For each such combination, it iterates over the Cartesian product of candidate 
        replacements for these nodes. The resulting new path is accepted if:
        - Every consecutive pair of nodes in the new path is adjacent in the graph.
        - The path remains simple (i.e. no duplicate nodes).
        
        Parameters:
        graph: An adjacency dictionary (mapping node -> set(neighbors)) or a NetworkX graph.
        path (list): A list of nodes representing the original simple path.
        npert (int): Maximum number of perturbed paths to generate.
        
        Returns:
        List[List]: A list of perturbed paths (each a list of nodes) up to npert paths.
        """
        perturbations = []
        # Determine the internal indices (excluding the endpoints)
        internal_indices = list(range(1, len(path) - 1))
        # Helper: get neighbors for a given node
        def get_neighbors(node):
            if hasattr(graph, "get"):  # assume it's a dict
                return graph.get(node, set())
            else:  # assume it's a NetworkX graph
                return set(graph[node])
        # Helper: check if two nodes are adjacent in the graph
        def are_adjacent(u, v):
            return v in get_neighbors(u)
        # Iterate over how many nodes to perturb simultaneously (at least 1)
        for r in range(1, len(internal_indices) + 1):
            # For each combination (subset) of internal indices to perturb
            for indices_to_perturb in itertools.combinations(internal_indices, r):
                # For each index selected for perturbation, compute candidate set based on the original path:
                candidate_lists = []
                valid_subset = True
                for i in indices_to_perturb:
                    prev_node = path[i - 1]
                    next_node = path[i + 1]
                    # Candidate set: nodes adjacent to both neighbors, excluding the original
                    candidates = get_neighbors(prev_node).intersection(get_neighbors(next_node))
                    candidates = set(candidates) - {path[i]}
                    if not candidates:
                        # If no candidate is available for this index, skip this combination.
                        valid_subset = False
                        break
                    candidate_lists.append(list(candidates))
                if not valid_subset:
                    continue
                # Iterate over all combinations of candidate replacements for these indices.
                for candidate_combo in itertools.product(*candidate_lists):
                    # Create a new path by copying the original path.
                    new_path = path.copy()
                    # Replace each selected internal node with its candidate from candidate_combo.
                    for idx, pos in enumerate(indices_to_perturb):
                        new_path[pos] = candidate_combo[idx]
                    # Verify that the new path is valid: each consecutive pair must be adjacent.
                    valid_path = True
                    for i in range(len(new_path) - 1):
                        if not are_adjacent(new_path[i], new_path[i + 1]):
                            valid_path = False
                            break
                    # Also check the path is simple (all nodes are unique).
                    if valid_path and len(set(new_path)) == len(new_path):
                        perturbations.append(new_path)
                        if len(perturbations) >= npert:
                            return perturbations
        return perturbations
    
    @staticmethod
    def __differentiate_regress(arr, axis=-1, msi_min=None, msi_max=None, normalize=True):
        """
        Compute the derivative of `arr` along `axis` after subtracting
        a fixed-slope line from each 1D slice along that axis.
        Optionally normalize gradient to [0, 1] based on theoretical max.

        Parameters
        ----------
        arr : ndarray
            N-dimensional data array.
        axis : int, default −1
            Axis along which to subtract slope and differentiate.
        msi_min, msi_max : float, optional
            Value bounds. If None, computed from data.
        normalize : bool, default True
            Whether to normalize gradient to [0, 1].

        Returns
        -------
        grad : ndarray
            Gradient of residuals (optionally normalized to [0, 1]).
        """
        # Defaults from data if not provided
        if msi_min is None:
            msi_min = arr.min()
        if msi_max is None:
            msi_max = arr.max()

        # Move target axis to end
        arr_rolled = np.moveaxis(arr, axis, -1)
        leading_shape = arr_rolled.shape[:-1]
        N = arr_rolled.shape[-1]
        x = np.arange(N)

        # Flatten leading dims
        arr_flat = arr_rolled.reshape(-1, N)

        # Fixed slope & intercept for regression line
        slope = (msi_max - msi_min) / float(N)
        intercept = msi_min  # keeps residual mean-zero for a perfect ramp
        y_hat = intercept + slope * x

        # Subtract regression line
        residuals_flat = arr_flat - y_hat[np.newaxis, :]

        # Reshape back to original shape
        residuals = residuals_flat.reshape(*leading_shape, N)
        residuals = np.moveaxis(residuals, -1, axis)

        # Compute gradient & scale by 1/N
        grad = np.gradient(residuals, x, axis=axis) * (1.0 / N)

        if normalize:
            Δ = msi_max - msi_min
            g_max = Δ * (N + 1) / (N**2)  # theoretical bound
            g_min = -g_max
            grad = (grad - g_min) / (g_max - g_min)  # maps [-g_max, g_max] → [0, 1]

        return grad



    @staticmethod
    def differentiate_regress(arr, axis=-1, msi_min = None, msi_max=None,regress=False):
        """
        Compute the derivative of `arr` w.r.t. `x` along `axis`, after subtracting
        a fixed‐slope “regression” line from each 1D slice along that axis.   
        Returns
        -------
        grad_scaled : ndarray
            The gradient ∂/∂x of [arr − (msi_max/N)·x], scaled by 1/N, with the same
            shape as `arr`.
        
        Parameters
        ----------
        arr : ndarray
            N-dimensional data array.  We will “regress out” a fixed slope along `axis`.
        x : 1D array of length N
            The coordinate values along the regression/differentiation axis.
        axis : int, default −1
            The axis of `arr` along which to subtract the fixed‐slope line and then differentiate.
        msi_max : float, default 255
            The “maximum” value used to define the slope as (msi_max / N).
        """
        if msi_min is None:
            msi_min = arr.min() 
        if msi_max is None:
            msi_max = arr.max() 
        # 1) Move the target axis to the end, so arr_rolled.shape == (..., N)
        arr_rolled = np.moveaxis(arr, axis, -1)
        leading_shape = arr_rolled.shape[:-1]  # shape of all dims except the last
        N = arr_rolled.shape[-1]               # number of samples along regression axis
        x = np.arange(N)
        # 2) Flatten the leading dims: arr_flat.shape = (K, N), where K = product(leading_shape)
        arr_flat = arr_rolled.reshape(-1, N)
        # 3) Compute the fixed slope = msi_max / N, then predict y_hat = slope * x
        slope = (msi_max-msi_min) / float(N)
        y_hat = slope * x                        # shape (N,)
        # 4) Subtract y_hat from each row of arr_flat to get residuals_flat
        #    Broadcasting: (K, N) - (1, N) → (K, N)
        if regress:
            residuals_flat = arr_flat - y_hat[np.newaxis, :]
        else:
            residuals_flat = arr_flat

        # 5) Reshape back to (..., N) and move the axis back:
        residuals = residuals_flat.reshape(*leading_shape, N)
        residuals = np.moveaxis(residuals, -1, axis)
        # 6) Differentiate the residuals along `axis` and scale by 1/N
        grad = np.gradient(residuals, x, axis=axis)
        return grad * (1.0 / N)

    def shannon_entropy(self,arr):
        """
        Compute Shannon entropy for each 1D slice of `arr`. If `arr` is 1D, return a single float.
        If `arr` is 2D with shape (n, p), return a 1D array of length n, where each entry is the
        entropy of the corresponding row.
        """
        bins = self.nbins_shannon
        arr = np.asarray(arr)
        # Case 1: 1D input → compute and return a single entropy
        if arr.ndim == 1:
            counts, _edges = np.histogram(arr, bins=bins)
            counts = counts.astype(float)
            total = counts.sum()
            if total == 0:
                return 0.0
            p = counts[counts > 0] / total
            return -np.sum(p * np.log(p))/np.log(bins)
        # Case 2: 2D input → compute entropy for each row separately
        elif arr.ndim == 2:
            # Pre-allocate an array to hold n entropies
            n = arr.shape[0]
            entropies = np.empty(n, dtype=float)
            for i in range(n):
                # Recursively call on the i-th row (which is 1D)
                entropies[i] = self.shannon_entropy(arr[i])
            return entropies
        else:
            raise ValueError("`shannon_entropy` expects a 1D or 2D array, but got ndim = {}".format(arr.ndim))


    def nabla_max(self,nodal_msi_arr):
        n = len(nodal_msi_arr)
        xmin,xmax = nodal_msi_arr.min(),nodal_msi_arr.max()
        ymax      = np.where(np.arange(n) % 2 == 0, xmin, xmax)
        grad_norm = self.differentiate_regress(ymax)
        return (grad_norm**2).sum()

    def total_cost(self,nodal_msi_arr):
        nabla_arr = self.differentiate_regress(nodal_msi_arr)
        entropy   = self.shannon_entropy(nodal_msi_arr)
        nabla     = (nabla_arr**2).sum()/self.nabla_max(nodal_msi_arr)
        total     = nabla / entropy
        return total, nabla ,entropy



    def _process_single(self,args):
        path, MS_mode_dict = args
        try:
            nodal_ms_arr = np.array([MS_mode_dict[p] for p in path if p in MS_mode_dict])
            if nodal_ms_arr.size == 0:
                return None
            cost, gradient, entropy = self.total_cost(nodal_ms_arr)
            if np.isnan(cost):
                return None
            return cost, gradient, entropy
        except Exception:
            return None


    def process_paths(self,paths, MS_mode_dict,  n_jobs=32):
        """
        Process a list of paths in parallel to compute cost, gradient, entropy,
        with a rich progress bar that advances only when workers finish.
        """
        results = []
        with Pool(processes=n_jobs) as pool, Progress() as progress:
            task = progress.add_task("[cyan]Processing paths...", total=len(paths))
            # feed arguments to workers
            args_iter = ((path, MS_mode_dict) for path in paths)
            for res in pool.imap(self._process_single, args_iter, chunksize=100):
                if res is not None:
                    results.append(res)
                progress.advance(task)
        if results:
            costs, grads, ents = zip(*results)
        else:
            costs, grads, ents = [], [], []
        return np.array(list(costs)),  np.array(list(grads)),  np.array(list(ents))