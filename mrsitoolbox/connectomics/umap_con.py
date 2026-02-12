import numpy as np
import numba
from warnings import warn

try:
    import pandas as pd
    import networkx as nx
    import datashader as ds
    import datashader.transfer_functions as tf
    import datashader.bundling as bd
    import matplotlib.pyplot as plt
    import bokeh.plotting as bpl
    import bokeh.transform as btr
    import holoviews as hv
    import holoviews.operation.datashader as hd

except ImportError:
    warn(
        """The umap.plot package requires extra plotting libraries to be installed.
    You can install these via pip using

    pip install umap-learn[plot]

    or via conda using

     conda install pandas matplotlib datashader bokeh holoviews colorcet scikit-image
    """
    )
    raise ImportError(
        "umap.plot requires pandas matplotlib datashader bokeh holoviews scikit-image and colorcet to be "
        "installed"
    ) from None

import plotly.graph_objects as go

from warnings import warn




def _get_embedding(umap_object):
    if hasattr(umap_object, "embedding_"):
        return umap_object.embedding_
    elif hasattr(umap_object, "embedding"):
        return umap_object.embedding
    else:
        raise ValueError("Could not find embedding attribute of umap_object")


def plot_connectivty(con_output,
                     edge_color="gray",node_cmap="spectral",
                     node_size = 20,background="white",
                     scale_title="Metabolic Similarity"
                     ):
    nodes_pos   = con_output["nodes"]
    edges_pos   = con_output["edges"]
    node_names  = con_output["node_names"]
    node_values = con_output["scalar_map"]


    # Plot nodes with custom colormap
    # Create edges as separate traces
    # Plot edges
    edge_trace = go.Scatter(
        x=edges_pos.x.to_numpy(),
        y=edges_pos.y.to_numpy(),
        mode='markers',
        marker=dict(color=edge_color, size=1),
        hoverinfo='none'
    )

    # Plot nodes with your custom colormap
    node_trace = go.Scatter(
        x=nodes_pos[:, 0],
        y=nodes_pos[:, 1],
        mode='markers',
        marker=dict(
            color=node_values,
            colorscale=node_cmap,  # Use your custom colormap
            size=node_size,  # Adjust as needed
            showscale=True,
            colorbar=dict(title=scale_title)
        ),
        text=[f"{name}" for name in node_names],  # Hover text per node
        hovertemplate='<b>%{text}</b><extra></extra>',  # Display only node name
    )

    # Layout with black background
    layout = go.Layout(
        plot_bgcolor=background,
        paper_bgcolor=background,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return go.Figure(data=[edge_trace, node_trace], layout=layout)

def connectivity(
    umap_object,
    edge_bundling=None,
    tsne_embedding=None,
    hemisphere="both",
    node_names=None,
    use_dask=False,
    iterations=500,
    points=None,
):
    """Plot connectivity relationships of the underlying UMAP
    simplicial set data structure. Internally UMAP will make
    use of what can be viewed as a weighted graph. This graph
    can be plotted using the layout provided by UMAP as a
    potential diagnostic view of the embedding. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    over-plotting issues and provide options for plotting the
    points as well as using edge bundling for graph visualization.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.
    edge_bundling: string or None (optional, default None)
        The edge bundling method to use. Currently supported
        are None or 'hammer'. See the datashader docs
        on graph visualization for more details.
    use_dask: Whether to use dask to parallelize the computation.
        (optional, default False)
    accuracy: Number of entries in table for (optional, default 500)
    Returns: dictionnary containing ....
    -------
    result: 
    """
    output   = dict()
    if points is None:
        points   = _get_embedding(umap_object)
        
    point_df = pd.DataFrame(points, columns=("x", "y"))
        
    if tsne_embedding is None:
        coo_graph = umap_object.graph_.tocoo()
        edge_df = pd.DataFrame(
            np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
            columns=("source", "target", "weight"),
        )
    else:
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = umap_object.n_neighbors
        # Build k-NN on the t-SNE result
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(tsne_embedding)
        distances, indices = nbrs.kneighbors(tsne_embedding)

        # Build edge list
        rows, cols = [], []
        for i, neighbors in enumerate(indices):
            for j, neighbor in enumerate(neighbors):
                rows.append(i)
                cols.append(neighbor)

        # Build DataFrame similar to UMAP's edge_df
        weights = np.exp(-distances.flatten())  # Optional: use distances to weight edges
        edge_df = pd.DataFrame({
            "source": np.repeat(np.arange(len(tsne_embedding)), n_neighbors),
            "target": cols,
            "weight": weights
        })


    edge_df["source"] = edge_df.source.astype(np.int32)
    edge_df["target"] = edge_df.target.astype(np.int32)

    if hemisphere != "both":
        if node_names is not None:
            # ignore opposed hemisphere and include BS regions
            if hemisphere == "lh":
                valid_indices = [i for i, name in enumerate(node_names) if "rh" not in name]
            elif hemisphere == "rh":
                valid_indices = [i for i, name in enumerate(node_names) if "lh" not in name]
            else:
                raise ValueError(f"Invalid hemisphere option: {hemisphere}")

            # Create a mapping from old indices to new indices after filtering
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

            # Filter point_df
            # Filter point_df and points
            point_df = point_df.iloc[valid_indices].reset_index(drop=True)
            points = points[valid_indices]

            # Filter tsne_embedding if provided
            if tsne_embedding is not None:
                tsne_embedding = tsne_embedding[valid_indices]

            # Filter edge_df: keep only rows where both source and target are in valid_indices
            valid_indices_set = set(valid_indices)
            edge_df = edge_df[edge_df["source"].isin(valid_indices_set) & edge_df["target"].isin(valid_indices_set)]

            # Remap source and target indices to match filtered data
            edge_df["source"] = edge_df["source"].map(index_mapping)
            edge_df["target"] = edge_df["target"].map(index_mapping)
            edge_df = edge_df.dropna().astype({"source": int, "target": int}).reset_index(drop=True)
            # Filter node_names
            node_names = [node_names[i] for i in valid_indices]
        else:
            raise ValueError("node_names must be provided when hemisphere filtering is applied.")


    if edge_bundling is None:
        edges = bd.directly_connect_edges(point_df, edge_df, weight="weight")
    elif edge_bundling == "hammer":
        if len(node_names)>1000:
            warn(
                "Hammer edge bundling is expensive for large graphs (n>1000)!\n"
                "This may take a long time to compute!"
            )
        edges = bd.hammer_bundle(point_df, edge_df, weight="weight",
                                 use_dask=use_dask,
                                 iterations=iterations,
                                 )
    else:
        raise ValueError(f"{edge_bundling} is not a recognised bundling method")

    output["edge_df"]    = edge_df
    output["edges"]      = edges
    output["nodes"]      = points
    output["node_names"] = node_names
    output["scalar_map"] = tsne_embedding.squeeze()


    return output