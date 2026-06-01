
import networkx as nx
import numpy as np
from nilearn import plotting
import numpy as np
import os
from os.path import join, exists, split
from nilearn.plotting import plot_connectome

from ..tools.debug import Debug

debug = Debug()

class Brain3D:
    def __init__(self):
        pass

    def plot_parcel_nodes(self,centroids, markersize=1,colormap="plasma",labels=None):
        """
        Plots the rich club nodes on a 3D brain using nilearn.view_markers, with marker sizes proportional to their degree.
        
        :param binarized_matrix: 2D numpy array representing the binarized adjacency matrix.
        :param coordinates: Numpy array with shape (n_nodes, 3) representing the 3D coordinates of each node.
        :param rich_club_nodes: List of node indices corresponding to rich club nodes.
        """
        # Get the coordinates and degrees of the rich club nodes
        # Normalize degree sizes for better visualization (optional: scale factor)
        marker_sizes = np.ones(centroids.shape[0])*markersize
        view = plotting.view_markers(centroids, marker_size=marker_sizes, marker_color=colormap,marker_labels=labels)
        # Open the view in a browser
        view.open_in_browser()
        return view  # Return view object for further customization if needed
    

    @staticmethod
    def plot_significant_connectome(results_dict, df_atlas, output_file,
                                    alpha=0.05, node_size=60, node_color="crimson",
                                    edge_cmap="Reds", edge_threshold=None,
                                    display_mode="ortho", black_bg=True):
        """
        Render significant NBS edges on a glass brain using nilearn.plot_connectome.
        Parameters
        ----------
        results_dict : dict
            Output from `NBS.bct_corr`.
        df_atlas : pandas.DataFrame
            Atlas table providing MNI coordinates via columns
            ['XCoord(mm)', 'YCoord(mm)', 'ZCoord(mm)'].
        output_file : str or pathlib.Path
            File path where the rendered connectome image will be saved.
        alpha : float, optional
            Component-level significance threshold.
        node_size : int or sequence, optional
            Passed to nilearn's plot_connectome.
        node_color : color or sequence, optional
            Passed to nilearn's plot_connectome.
        edge_cmap : str, optional
            Colormap for edges.
        edge_threshold : float or str or None, optional
            Edge threshold forwarded to nilearn.
        display_mode : str, optional
            Display mode for nilearn glass brain.
        black_bg : bool, optional
            Whether to draw on a black background.

        Returns
        -------
        display : nilearn.plotting.displays.OrthoProjector or None
            Display object returned by nilearn (None if output_file provided).
        """
        comp_masks = results_dict["comp_masks"]
        comp_pvals = results_dict["comp_pvals"]
        t_mat = results_dict["t_mat"]
        n_nodes = t_mat.shape[0]
        if len(df_atlas) != n_nodes:
            raise ValueError("Atlas table length does not match adjacency size.")
        combined_mask = np.zeros_like(t_mat, dtype=bool)
        for mask, pval in zip(comp_masks, comp_pvals):
            if pval < alpha:
                combined_mask |= mask
        if not np.any(combined_mask):
            debug.warning("plot_significant_connectome: no significant edges to plot.")
            return None
        sig_adj = np.zeros_like(t_mat, dtype=float)
        sig_adj[combined_mask] = t_mat[combined_mask]
        sig_adj = np.triu(sig_adj, k=1)
        sig_adj = sig_adj + sig_adj.T
        active_nodes = np.where(sig_adj.sum(axis=0) != 0)[0]
        sig_adj = sig_adj[np.ix_(active_nodes, active_nodes)]
        node_coords = df_atlas.loc[active_nodes, ["XCoord(mm)", "YCoord(mm)", "ZCoord(mm)"]].to_numpy(dtype=float)
        if sig_adj.size == 0:
            debug.warning("plot_significant_connectome: significant mask empty after filtering.")
            return None
        display = plot_connectome(
            sig_adj,
            node_coords,
            node_size=node_size,
            node_color=node_color,
            edge_cmap=edge_cmap,
            edge_threshold=edge_threshold,
            output_file=output_file,
            display_mode=display_mode,
            black_bg=black_bg,
            annotate=False,
            colorbar=True,
        )
        return display
