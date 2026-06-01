
from ..tools.debug import Debug
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

    @staticmethod
    def graph_from_edge_pairs(edge_pairs, n_nodes=None):
        """
        Build an unweighted NetworkX graph from binary GM adjacency edge pairs.

        Parameters
        ----------
        edge_pairs : array-like, shape (n_edges, 2)
            Integer node-index pairs.
        n_nodes : int | None
            Optional node count. When provided, isolated nodes are preserved.
        """
        G = nx.Graph()
        if n_nodes is not None:
            G.add_nodes_from(range(int(n_nodes)))
        pairs = np.asarray(edge_pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[0] == 0:
            return G
        if pairs.shape[1] != 2:
            raise ValueError("edge_pairs must have shape (n_edges, 2).")
        for u, v in pairs:
            u = int(u)
            v = int(v)
            if u == v:
                continue
            if n_nodes is not None and (u < 0 or v < 0 or u >= int(n_nodes) or v >= int(n_nodes)):
                continue
            G.add_edge(u, v)
        return G

    @staticmethod
    def diffusion_embedding_positions(
        adjacency=None,
        *,
        edge_pairs=None,
        n_nodes=None,
        components=(2, 3),
        scale_by_eigenvalue=True,
    ):
        """
        Compute 2D node positions from diffusion components of a binary graph.

        ``components`` uses 1-based indices in the sorted diffusion spectrum. Thus
        ``(2, 3)`` uses the second and third diffusion components.
        """
        if edge_pairs is not None:
            graph = NetFibre.graph_from_edge_pairs(edge_pairs, n_nodes=n_nodes)
        elif isinstance(adjacency, nx.Graph):
            graph = adjacency.copy()
            if n_nodes is not None:
                graph.add_nodes_from(range(int(n_nodes)))
        elif isinstance(adjacency, dict):
            graph = NetFibre.construct_gm_graph(adjacency)
            if n_nodes is not None:
                graph.add_nodes_from(range(int(n_nodes)))
        elif adjacency is not None:
            arr = np.asarray(adjacency)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                graph = nx.from_numpy_array((arr > 0).astype(int))
                if n_nodes is not None:
                    graph.add_nodes_from(range(int(n_nodes)))
            elif arr.ndim == 2 and arr.shape[1] == 2:
                graph = NetFibre.graph_from_edge_pairs(arr, n_nodes=n_nodes)
            else:
                raise ValueError("adjacency must be a graph, adjacency dict, square matrix, or edge-pair array.")
        else:
            raise ValueError("Provide adjacency or edge_pairs.")

        node_order = sorted(int(node) for node in graph.nodes())
        if n_nodes is not None:
            for node in range(int(n_nodes)):
                if node not in graph:
                    graph.add_node(node)
            node_order = list(range(int(n_nodes)))
        if not node_order:
            return {}, np.zeros((0, 2), dtype=float), np.asarray([], dtype=int)

        matrix = nx.to_numpy_array(graph, nodelist=node_order, dtype=float)
        matrix = np.maximum((matrix > 0).astype(float), (matrix.T > 0).astype(float))
        n = int(matrix.shape[0])
        if n == 1:
            return {int(node_order[0]): (0.0, 0.0)}, np.zeros((1, 2), dtype=float), np.asarray(node_order, dtype=int)

        degrees = np.asarray(matrix.sum(axis=1), dtype=float)
        inv_sqrt_degree = np.zeros_like(degrees, dtype=float)
        positive = degrees > 0.0
        inv_sqrt_degree[positive] = 1.0 / np.sqrt(degrees[positive])
        diffusion_operator = inv_sqrt_degree[:, None] * matrix * inv_sqrt_degree[None, :]

        eigenvalues, eigenvectors = np.linalg.eigh(diffusion_operator)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.asarray(eigenvalues[order], dtype=float)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)

        coords = np.zeros((n, 2), dtype=float)
        requested = [max(1, int(component)) - 1 for component in tuple(components or (2, 3))]
        for out_idx, eig_idx in enumerate(requested[:2]):
            if eig_idx >= eigenvectors.shape[1]:
                continue
            vector = np.asarray(eigenvectors[:, eig_idx], dtype=float)
            if scale_by_eigenvalue and eig_idx < eigenvalues.shape[0]:
                vector = vector * float(eigenvalues[eig_idx])
            finite = np.isfinite(vector)
            if np.any(finite):
                pivot = int(np.nanargmax(np.abs(vector)))
                if vector[pivot] < 0:
                    vector = -vector
            coords[:, out_idx] = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        for dim in range(coords.shape[1]):
            values = coords[:, dim]
            span = float(np.nanmax(values) - np.nanmin(values)) if values.size else 0.0
            if np.isfinite(span) and span > 1e-12:
                coords[:, dim] = (values - float(np.nanmean(values))) / span

        positions = {
            int(node): (float(coords[idx, 0]), float(coords[idx, 1]))
            for idx, node in enumerate(node_order)
        }
        return positions, coords, np.asarray(node_order, dtype=int)

    @staticmethod
    def spiral_path_positions(path_nodes, *, turns=None, start_radius=0.12, end_radius=1.0):
        """
        Compute sequence-preserving 2D positions along an Archimedean spiral.

        The node order is taken directly from ``path_nodes``.
        """
        nodes = [] if path_nodes is None else [int(node) for node in list(path_nodes)]
        if not nodes:
            return {}, np.zeros((0, 2), dtype=float)
        if len(nodes) == 1:
            return {int(nodes[0]): (0.0, 0.0)}, np.zeros((1, 2), dtype=float)

        if turns is None:
            turns = max(1.0, min(4.5, float(len(nodes)) / 8.0))
        try:
            turns = max(0.25, float(turns))
        except Exception:
            turns = max(1.0, min(4.5, float(len(nodes)) / 8.0))
        try:
            start_radius = max(0.0, float(start_radius))
        except Exception:
            start_radius = 0.12
        try:
            end_radius = max(start_radius + 1e-6, float(end_radius))
        except Exception:
            end_radius = 1.0

        theta = np.linspace(0.0, 2.0 * np.pi * turns, len(nodes), dtype=float)
        radius = np.linspace(start_radius, end_radius, len(nodes), dtype=float)
        coords = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        positions = {
            int(node): (float(coords[idx, 0]), float(coords[idx, 1]))
            for idx, node in enumerate(nodes)
        }
        return positions, coords

    @staticmethod
    def _branch_spiral_positions(branch_nodes, main_positions, *, branch_scale=0.48, turns=None):
        nodes = [] if branch_nodes is None else [int(node) for node in list(branch_nodes)]
        if not nodes:
            return {}, np.zeros((0, 2), dtype=float)
        if turns is None:
            turns = max(0.5, min(2.0, float(len(nodes)) / 8.0))
        try:
            turns = max(0.25, float(turns))
        except Exception:
            turns = max(0.5, min(2.0, float(len(nodes)) / 8.0))
        try:
            branch_scale = max(0.05, float(branch_scale))
        except Exception:
            branch_scale = 0.48

        anchor_idx = 0
        for idx, node in enumerate(nodes):
            if int(node) in main_positions:
                anchor_idx = idx
                break
        anchor_node = int(nodes[anchor_idx])
        anchor_pos = np.asarray(main_positions.get(anchor_node, (0.0, 0.0)), dtype=float)
        radial_angle = float(np.arctan2(anchor_pos[1], anchor_pos[0]))
        if not np.isfinite(radial_angle):
            radial_angle = 0.0

        ordered_nodes = nodes[anchor_idx:]
        if len(ordered_nodes) < 2:
            return {anchor_node: (float(anchor_pos[0]), float(anchor_pos[1]))}, anchor_pos.reshape(1, 2)

        theta = radial_angle + np.linspace(0.0, 2.0 * np.pi * turns, len(ordered_nodes), dtype=float)
        radius = np.linspace(0.0, branch_scale, len(ordered_nodes), dtype=float)
        rel = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        coords = anchor_pos.reshape(1, 2) + rel
        positions = {
            int(node): (float(coords[idx, 0]), float(coords[idx, 1]))
            for idx, node in enumerate(ordered_nodes)
        }
        return positions, coords

    @staticmethod
    def _draw_curved_edges(
        ax,
        positions,
        edges,
        *,
        color="black",
        width=2.2,
        alpha=0.9,
        curvature=0.18,
        zorder=1,
        direction=1.0,
        linestyle="solid",
    ):
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        try:
            direction = 1.0 if float(direction) >= 0.0 else -1.0
        except Exception:
            direction = 1.0

        for u, v in list(edges or []):
            if int(u) not in positions or int(v) not in positions:
                continue
            p0 = np.asarray(positions[int(u)], dtype=float)
            p1 = np.asarray(positions[int(v)], dtype=float)
            delta = p1 - p0
            length = float(np.linalg.norm(delta))
            if not np.isfinite(length) or length <= 1e-12:
                continue
            normal = np.asarray([-delta[1], delta[0]], dtype=float) / length
            control = 0.5 * (p0 + p1) + direction * float(curvature) * length * normal
            path = Path([p0, control, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=float(width),
                alpha=float(alpha),
                capstyle="round",
                joinstyle="round",
                linestyle=str(linestyle),
                zorder=zorder,
            )
            ax.add_patch(patch)

    @staticmethod
    def plot_fibrenet_path(
        path_nodes,
        *,
        adjacency=None,
        edge_pairs=None,
        n_nodes=None,
        node_names=None,
        node_colors=None,
        ax=None,
        title=None,
        components=(2, 3),
        layout="diffusion",
        edge_color="black",
        edge_width=2.2,
        node_size=280,
        font_size=8,
        spiral_turns=None,
        branch_paths=None,
        branch_node_names=None,
        branch_node_colors=None,
        branch_edge_color="0.25",
        branch_edge_width=None,
        branch_node_size=None,
        endpoint_node_size=None,
        spiral_curved_edges=True,
        spiral_edge_curvature=0.18,
        branch_edge_linestyle="solid",
    ):
        """
        Plot a path as a FibreNet graph.

        Only consecutive nodes in ``path_nodes`` are connected. With
        ``layout="diffusion"``, positions are computed from the full binary GM
        adjacency. With ``layout="spiral"``, positions follow the path sequence.
        """
        import matplotlib.pyplot as plt

        nodes = [] if path_nodes is None else [int(node) for node in list(path_nodes)]
        if len(nodes) < 1:
            raise ValueError("path_nodes must contain at least one node.")
        inferred_n = max(nodes) + 1
        if n_nodes is None:
            if edge_pairs is not None:
                pairs = np.asarray(edge_pairs, dtype=int)
                if pairs.ndim == 2 and pairs.size:
                    inferred_n = max(inferred_n, int(np.max(pairs)) + 1)
            n_nodes = inferred_n

        layout_name = str(layout or "diffusion").strip().lower()
        if layout_name in {"spiral", "path_spiral", "sequence", "sequence_spiral"}:
            layout_name = "spiral"
            positions, _coords = NetFibre.spiral_path_positions(nodes, turns=spiral_turns)
            branch_positions = {}
            for branch_path in list(branch_paths or []):
                branch_pos, _branch_coords = NetFibre._branch_spiral_positions(
                    branch_path,
                    positions,
                    branch_scale=0.46,
                )
                branch_positions.update(branch_pos)
            positions.update(branch_positions)
        else:
            layout_name = "diffusion"
            positions, _coords, _node_order = NetFibre.diffusion_embedding_positions(
                adjacency,
                edge_pairs=edge_pairs,
                n_nodes=n_nodes,
                components=components,
            )

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((int(a), int(b)) for a, b in zip(nodes[:-1], nodes[1:]))
        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 6.0))
        pos = {
            int(node): positions.get(int(node), (float(idx), 0.0))
            for idx, node in enumerate(nodes)
        }
        colors = [] if node_colors is None else list(node_colors)
        if len(colors) != len(nodes):
            colors = ["0.7"] * len(nodes)
        labels_source = [] if node_names is None else list(node_names)
        labels = {
            int(node): str(labels_source[idx]) if idx < len(labels_source) else str(int(node))
            for idx, node in enumerate(nodes)
        }

        main_edges = list(zip(nodes[:-1], nodes[1:]))
        if layout_name == "spiral" and spiral_curved_edges:
            NetFibre._draw_curved_edges(
                ax,
                positions,
                main_edges,
                color=edge_color,
                width=float(edge_width),
                alpha=0.9,
                curvature=spiral_edge_curvature,
                zorder=1,
                direction=1.0,
            )
        else:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=main_edges,
                ax=ax,
                edge_color=edge_color,
                width=float(edge_width),
                alpha=0.9,
            )
        default_endpoint_size = float(node_size) * 1.65 if layout_name == "spiral" else float(node_size)
        endpoint_size = float(endpoint_node_size if endpoint_node_size is not None else default_endpoint_size)
        node_sizes = [float(node_size)] * len(nodes)
        if len(node_sizes) == 1:
            node_sizes[0] = endpoint_size
        elif len(node_sizes) > 1:
            node_sizes[0] = endpoint_size
            node_sizes[-1] = endpoint_size
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=colors,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=0.8,
            ax=ax,
        )
        all_labels = dict(labels)

        branch_paths = list(branch_paths or [])
        branch_names = list(branch_node_names or [])
        branch_colors = list(branch_node_colors or [])
        for branch_idx, branch_path in enumerate(branch_paths):
            branch_nodes = [int(node) for node in list(branch_path or [])]
            if len(branch_nodes) < 2:
                continue
            branch_pos = {
                int(node): positions[int(node)]
                for node in branch_nodes
                if int(node) in positions
            }
            branch_graph = nx.Graph()
            branch_graph.add_nodes_from(branch_nodes)
            branch_edges = list(zip(branch_nodes[:-1], branch_nodes[1:]))
            branch_graph.add_edges_from(branch_edges)
            if layout_name == "spiral" and spiral_curved_edges:
                NetFibre._draw_curved_edges(
                    ax,
                    positions,
                    branch_edges,
                    color=branch_edge_color,
                    width=float(branch_edge_width if branch_edge_width is not None else max(0.8, float(edge_width) * 0.72)),
                    alpha=0.82,
                    curvature=float(spiral_edge_curvature) * 1.35,
                    zorder=2,
                    direction=1.0,
                    linestyle=branch_edge_linestyle,
                )
            else:
                nx.draw_networkx_edges(
                    branch_graph,
                    branch_pos,
                    edgelist=branch_edges,
                    ax=ax,
                    edge_color=branch_edge_color,
                    width=float(branch_edge_width if branch_edge_width is not None else max(0.8, float(edge_width) * 0.72)),
                    alpha=0.82,
                    style=str(branch_edge_linestyle),
                )
            branch_draw_nodes = [
                int(node)
                for node in branch_nodes
                if int(node) not in set(nodes) and int(node) in positions
            ]
            local_colors_all = branch_colors[branch_idx] if branch_idx < len(branch_colors) else []
            local_colors_all = list(local_colors_all or [])
            if len(local_colors_all) != len(branch_nodes):
                local_colors_all = ["0.62"] * len(branch_nodes)
            node_color_lookup = {
                int(node): local_colors_all[idx]
                for idx, node in enumerate(branch_nodes)
            }
            local_colors = [node_color_lookup[int(node)] for node in branch_draw_nodes]
            local_size = float(branch_node_size if branch_node_size is not None else max(70.0, float(node_size) * 0.62))
            local_sizes = [local_size] * len(branch_draw_nodes)
            if local_sizes:
                for idx, node in enumerate(branch_draw_nodes):
                    if int(node) == int(branch_nodes[-1]):
                        local_sizes[idx] = max(local_size * 1.55, endpoint_size * 0.7)
            if branch_draw_nodes:
                nx.draw_networkx_nodes(
                    branch_graph,
                    branch_pos,
                    nodelist=branch_draw_nodes,
                    node_color=local_colors,
                    node_size=local_sizes,
                    edgecolors="black",
                    linewidths=0.7,
                    ax=ax,
                )
            local_names = branch_names[branch_idx] if branch_idx < len(branch_names) else []
            local_names = list(local_names or [])
            for idx, node in enumerate(branch_nodes):
                if node not in all_labels:
                    all_labels[node] = str(local_names[idx]) if idx < len(local_names) else str(node)

        label_pos = {
            int(node): positions[int(node)]
            for node in all_labels
            if int(node) in positions
        }
        nx.draw_networkx_labels(
            graph,
            label_pos,
            labels=all_labels,
            font_size=int(font_size),
            font_color="black",
            ax=ax,
        )
        if title:
            ax.set_title(str(title), fontsize=11)
        if layout_name == "spiral":
            ax.set_axis_off()
        else:
            ax.set_xlabel("GM diffusion component 2")
            ax.set_ylabel("GM diffusion component 3")
        ax.set_aspect("equal", adjustable="datalim")
        if layout_name != "spiral":
            ax.grid(True, alpha=0.18)
        return ax, positions

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
