
import numpy as np
from ..tools.debug import Debug

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
from itertools import combinations




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

    def compute_adjacency_list(
        self,
        label_image,
        labels=None,
        parcel_names=None,
        order=1,
        return_matrix=False,
    ):
        """
        Compute parcel adjacency from a 3D label image using 6-connectivity.

        Parameters
        ----------
        label_image : array-like
            3D parcel label image.
        labels : array-like, optional
            Labels to include. If None, all non-zero labels are used.
        parcel_names : list[str], optional
            Parcel names aligned to sorted labels. If provided, direct
            cerebellum-cortex links are skipped.
        order : int, optional
            Neighbor order. `1` keeps direct neighbors only; values >1 include
            higher-order neighbors by graph expansion.
        return_matrix : bool, optional
            If True, return adjacency matrix in the order of `labels`.

        Returns
        -------
        dict or ndarray
            Adjacency dictionary `{label: set(neighbors)}` or adjacency matrix.
        """
        arr = np.asarray(label_image).astype(int)
        if arr.ndim != 3:
            raise ValueError(f"label_image must be 3D, got shape {arr.shape}.")

        if labels is None:
            labels = np.unique(arr)
        labels = np.asarray(labels).astype(int)
        labels = labels[labels != 0]
        labels_sorted = np.array(sorted([int(v) for v in labels]), dtype=int)
        label_set = set(int(v) for v in labels_sorted.tolist())

        if parcel_names is not None:
            if len(parcel_names) != len(labels_sorted):
                raise ValueError("Length of parcel_names must match number of labels.")
            label_to_name = {
                int(lbl): str(name) for lbl, name in zip(labels_sorted.tolist(), parcel_names)
            }
        else:
            label_to_name = None

        adjacency_dict = {int(label): set() for label in labels_sorted.tolist()}

        neighbor_offsets = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]

        label_image_padded = np.pad(arr, pad_width=1, mode="constant", constant_values=0)
        current_labels = label_image_padded[1:-1, 1:-1, 1:-1]

        for offset in neighbor_offsets:
            shifted_label_image = np.roll(label_image_padded, shift=offset, axis=(0, 1, 2))
            x_shift, y_shift, z_shift = offset

            if x_shift > 0:
                shifted_label_image[: x_shift + 1, :, :] = 0
            elif x_shift < 0:
                shifted_label_image[x_shift - 1 :, :, :] = 0
            if y_shift > 0:
                shifted_label_image[:, : y_shift + 1, :] = 0
            elif y_shift < 0:
                shifted_label_image[:, y_shift - 1 :, :] = 0
            if z_shift > 0:
                shifted_label_image[:, :, : z_shift + 1] = 0
            elif z_shift < 0:
                shifted_label_image[:, :, z_shift - 1 :] = 0

            neighbor_labels = shifted_label_image[1:-1, 1:-1, 1:-1]
            diff = (current_labels != neighbor_labels) & (current_labels != 0) & (neighbor_labels != 0)
            if not np.any(diff):
                continue
            label_pairs = np.stack([current_labels[diff], neighbor_labels[diff]], axis=1)

            for label1, label2 in label_pairs:
                label1 = int(label1)
                label2 = int(label2)
                if label1 not in label_set or label2 not in label_set:
                    continue
                if label_to_name is not None:
                    name1 = label_to_name.get(label1, "")
                    name2 = label_to_name.get(label2, "")
                    if ("cer-" in name1 and "ctx-" in name2) or ("ctx-" in name1 and "cer-" in name2):
                        continue
                adjacency_dict[label1].add(label2)
                adjacency_dict[label2].add(label1)

        if int(order) >= 2:
            expanded = {label: set(neighbors) for label, neighbors in adjacency_dict.items()}
            for _ in range(1, int(order)):
                for label, neighbors in expanded.items():
                    higher = set()
                    for neighbor in neighbors:
                        higher.update(adjacency_dict.get(int(neighbor), set()))
                    higher.discard(label)
                    higher.difference_update(neighbors)
                    if label_to_name is not None:
                        name1 = label_to_name.get(label, "")
                        filtered = set()
                        for second_neighbor in higher:
                            name2 = label_to_name.get(int(second_neighbor), "")
                            if not (
                                ("cer-" in name1 and "ctx-" in name2)
                                or ("ctx-" in name1 and "cer-" in name2)
                            ):
                                filtered.add(int(second_neighbor))
                        higher = filtered
                    adjacency_dict[label].update(higher)

        if return_matrix:
            return self.graphdict_to_mat(adjacency_dict, labels_order=labels_sorted.tolist())
        return adjacency_dict

    @staticmethod
    def _normalize_label(label):
        if isinstance(label, np.generic):
            return label.item()
        return label

    @staticmethod
    def mat_to_graphdict(adjacency_mat, labels=None, symmetric=True):
        """Convert adjacency matrix to adjacency-dict {label: [neighbor_labels]}."""
        mat = np.asarray(adjacency_mat)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("adjacency_mat must be a square 2D array.")
        n = mat.shape[0]
        if labels is None:
            labels = list(range(n))
        labels = [NetTools._normalize_label(lbl) for lbl in labels]
        if len(labels) != n:
            raise ValueError("labels length must match adjacency_mat shape.")

        adj = {lbl: set() for lbl in labels}
        for i in range(n):
            li = labels[i]
            for j in range(n):
                if i == j:
                    continue
                if mat[i, j] != 0:
                    lj = labels[j]
                    adj[li].add(lj)
                    if symmetric:
                        adj[lj].add(li)
        return {k: sorted(v) for k, v in adj.items()}

    @staticmethod
    def merge_adjacency_dicts(*adjacency_dicts):
        """Merge multiple adjacency dicts by union of neighbors."""
        merged = {}
        for adj in adjacency_dicts:
            if adj is None:
                continue
            for node, neighbors in adj.items():
                node_key = NetTools._normalize_label(node)
                merged.setdefault(node_key, set())
                for nb in neighbors:
                    merged[node_key].add(NetTools._normalize_label(nb))
        return {k: sorted(v) for k, v in merged.items()}

    @staticmethod
    def reorder_adjacency_dict(adj_dict, labels_order):
        """Restrict/reorder adjacency dict to a provided node order."""
        order = [NetTools._normalize_label(lbl) for lbl in labels_order]
        order_set = set(order)
        out = {}
        for node in order:
            neighbors = adj_dict.get(node, [])
            out[node] = sorted(
                NetTools._normalize_label(nb)
                for nb in neighbors
                if NetTools._normalize_label(nb) in order_set and NetTools._normalize_label(nb) != node
            )
        return out

    @staticmethod
    def graphdict_to_mat(adj_dict, labels_order=None, dtype=float):
        """Convert adjacency dict back to matrix (0/1 by default)."""
        if labels_order is None:
            labels_order = sorted(adj_dict.keys())
        labels = [NetTools._normalize_label(lbl) for lbl in labels_order]
        idx = {lbl: i for i, lbl in enumerate(labels)}
        mat = np.zeros((len(labels), len(labels)), dtype=dtype)
        for node, neighbors in adj_dict.items():
            node = NetTools._normalize_label(node)
            if node not in idx:
                continue
            i = idx[node]
            for nb in neighbors:
                nb = NetTools._normalize_label(nb)
                if nb in idx and nb != node:
                    j = idx[nb]
                    mat[i, j] = 1
        return mat

   

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



    def dimreduce_matrix(self, data, method='pca_tsne', scale_factor=255.0, 
                         output_dim=1,perplexity=30,norm=False):
        """
        Project a 4D array onto a 1D array using specified manifold learning method.
        
        Parameters:
        data (np.ndarray): Input array
        method (str): Method to use for projection. Options are:
                    'pca_tsne', 'umap', 'diffusion'
        scale_factor (float): Scale factor to multiply the final projection.
        output_dim (int): Number of dimensions to project to. Returns the nth component
                        if output_dim=n.
        
        Returns:
        np.ndarray: Output array of shape (N,) after projection, containing only the nth component.
        """
        if method == 'pca_tsne':
            premodel = PCA(n_components=50)
            model = TSNE(n_components=output_dim, method="exact", perplexity=perplexity)
            data = premodel.fit_transform(data)
            transformed_data = model.fit_transform(data)
        elif method == 'umap':
            from umap import UMAP
            premodel = PCA(n_components=50)
            model = UMAP(n_components=output_dim)
            data = premodel.fit_transform(data)
            transformed_data = model.fit_transform(data)
        elif method in ('diffusion', 'diffusion_embedding', 'dm'):
            from brainspace.gradient import GradientMaps
            data = np.asarray(data, dtype=float)
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError("Diffusion embedding expects a square (n x n) similarity matrix.")
            # Rescale MeSiM similarity to [0, 1] for diffusion embedding.
            data = (data + 1.0) / 2.0
            data = np.clip(data, 0.0, 1.0)
            gm = GradientMaps(n_components=output_dim, approach="dm", kernel="normalized_angle")
            gm.fit(data)
            transformed_data = gm.gradients_
        else:
            raise ValueError("Invalid method specified. Choose from 'pca_tsne', 'umap', 'diffusion'.")

        # Normalize data
        if norm:
            transformed_data -= np.nanmin(transformed_data, axis=0)
            denom = np.nanmax(transformed_data, axis=0)
            denom = np.where(denom == 0, 1.0, denom)
            transformed_data /= denom
        if method == 'umap':
            transformed_data -= np.nanmax(transformed_data, axis=0)
            transformed_data *= -1

        # Select the nth component and scale
        nth_component = transformed_data[:, output_dim - 1]  # Adjust for 0-based indexing
        return nth_component * scale_factor

    @staticmethod
    def _normalize_triangular_color_order(value, default="RBG"):
        text = str(value or default).strip().upper()
        valid = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"}
        if text not in valid:
            text = str(default or "RBG").strip().upper()
        if text not in valid:
            text = "RBG"
        return text

    @staticmethod
    def _triangular_rgb_bounds(x_values, y_values):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        x_min, x_max = np.nanmin(x_valid), np.nanmax(x_valid)
        y_min, y_max = np.nanmin(y_valid), np.nanmax(y_valid)
        return float(x_min), float(x_max), float(y_min), float(y_max)

    @staticmethod
    def _fallback_triangle_vertices(x_values, y_values):
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite_mask):
            return np.array([[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=float)
        x_valid = x_values[finite_mask]
        y_valid = y_values[finite_mask]
        x_min, x_max, y_min, y_max = NetTools._triangular_rgb_bounds(x_valid, y_valid)
        return np.array(
            [
                [0.5 * (x_min + x_max), y_max],
                [x_min, y_min],
                [x_max, y_min],
            ],
            dtype=float,
        )

    @staticmethod
    def _triangle_area2(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @staticmethod
    def fit_triangle_vertices(x_values, y_values):
        points = np.column_stack(
            (
                np.asarray(x_values, dtype=float).reshape(-1),
                np.asarray(y_values, dtype=float).reshape(-1),
            )
        )
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if points.shape[0] < 3:
            return NetTools._fallback_triangle_vertices(x_values, y_values)

        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] < 3:
            return NetTools._fallback_triangle_vertices(x_values, y_values)

        candidate_points = unique_points
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(unique_points)
            candidate_points = unique_points[hull.vertices]
        except Exception:
            pass

        if candidate_points.shape[0] > 96:
            step = max(1, int(np.ceil(candidate_points.shape[0] / 96.0)))
            reduced = candidate_points[::step]
            extrema = np.unique(
                np.vstack(
                    (
                        candidate_points[np.argmin(candidate_points[:, 0])],
                        candidate_points[np.argmax(candidate_points[:, 0])],
                        candidate_points[np.argmin(candidate_points[:, 1])],
                        candidate_points[np.argmax(candidate_points[:, 1])],
                    )
                ),
                axis=0,
            )
            candidate_points = np.unique(np.vstack((reduced, extrema)), axis=0)

        best_vertices = None
        best_area = -1.0
        for idx_a, idx_b, idx_c in combinations(range(candidate_points.shape[0]), 3):
            a = candidate_points[idx_a]
            b = candidate_points[idx_b]
            c = candidate_points[idx_c]
            area2 = NetTools._triangle_area2(a, b, c)
            if area2 > best_area:
                best_area = area2
                best_vertices = np.asarray((a, b, c), dtype=float)

        if best_vertices is None or best_area <= 1e-8:
            return NetTools._fallback_triangle_vertices(x_values, y_values)

        apex_index = int(np.argmax(best_vertices[:, 1]))
        apex = best_vertices[apex_index]
        base = np.delete(best_vertices, apex_index, axis=0)
        base = base[np.argsort(base[:, 0])]
        return np.asarray((apex, base[0], base[1]), dtype=float)

    @staticmethod
    def _normalize_rgb_chroma(values):
        colors = np.asarray(values, dtype=float)
        if colors.ndim != 2 or colors.shape[1] != 3:
            return np.clip(colors, 0.0, 1.0)
        colors = np.clip(colors, 0.0, 1.0)
        scale = np.max(colors, axis=1, keepdims=True)
        scale[scale <= 1e-9] = 1.0
        return np.clip(colors / scale, 0.0, 1.0)

    def triangular_rgb_model(self, x_values, y_values, color_order="RBG"):
        order = self._normalize_triangular_color_order(color_order)
        rgb_basis = {
            "R": np.array((1.0, 0.0, 0.0), dtype=float),
            "G": np.array((0.0, 1.0, 0.0), dtype=float),
            "B": np.array((0.0, 0.0, 1.0), dtype=float),
        }
        vertices = self.fit_triangle_vertices(x_values, y_values)
        return {
            "vertices": np.asarray(vertices, dtype=float),
            "anchor_points": np.asarray(vertices, dtype=float),
            "vertex_colors": np.asarray([rgb_basis[channel] for channel in order], dtype=float),
            "order": order,
            "fit_mode": "triangle",
        }

    def triangular_rgb_weights_from_model(self, x_values, y_values, model):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        weights_full = np.zeros((x_valid.shape[0], 3), dtype=float)
        finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
        if not np.any(finite_mask):
            return weights_full

        vertices = np.asarray(model["vertices"], dtype=float)
        v0, v1, v2 = vertices
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if np.isclose(denom, 0.0):
            fallback = self.triangular_rgb_model(
                x_valid[finite_mask],
                y_valid[finite_mask],
                model.get("order", "RBG"),
            )
            vertices = np.asarray(fallback["vertices"], dtype=float)
            v0, v1, v2 = vertices
            denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
            if np.isclose(denom, 0.0):
                return weights_full

        points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
        w0 = (
            (v1[1] - v2[1]) * (points[:, 0] - v2[0])
            + (v2[0] - v1[0]) * (points[:, 1] - v2[1])
        ) / denom
        w1 = (
            (v2[1] - v0[1]) * (points[:, 0] - v2[0])
            + (v0[0] - v2[0]) * (points[:, 1] - v2[1])
        ) / denom
        w2 = 1.0 - w0 - w1
        weights = np.column_stack((w0, w1, w2))
        weights = np.clip(weights, 0.0, 1.0)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum[weight_sum <= 0] = 1.0
        weights /= weight_sum
        weights_full[finite_mask] = weights
        return weights_full

    def triangular_rgb_colors_from_model(self, x_values, y_values, model):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        colors = np.full((x_valid.shape[0], 3), 0.65, dtype=float)
        finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
        if not np.any(finite_mask):
            return colors
        vertex_colors = np.asarray(model["vertex_colors"], dtype=float)
        weights = self.triangular_rgb_weights_from_model(x_values, y_values, model)
        colors[finite_mask] = self._normalize_rgb_chroma(weights[finite_mask] @ vertex_colors)
        return np.clip(colors, 0.0, 1.0)

    def triangular_rgb_scalar_from_model(
        self,
        x_values,
        y_values,
        model,
        *,
        channel_scalar_map=None,
    ):
        if channel_scalar_map is None:
            channel_scalar_map = {"R": 0.0, "B": 0.5, "G": 1.0}
        scalar_map = {
            str(key).strip().upper(): float(value)
            for key, value in dict(channel_scalar_map or {}).items()
            if str(key).strip()
        }
        order = [str(channel).strip().upper() for channel in model.get("order", "RBG")]
        vertex_scalars = np.asarray([scalar_map.get(channel, float(idx)) for idx, channel in enumerate(order)], dtype=float)
        weights = self.triangular_rgb_weights_from_model(x_values, y_values, model)
        scalar_values = np.full(weights.shape[0], np.nan, dtype=float)
        valid_mask = np.sum(weights, axis=1) > 0.0
        if np.any(valid_mask):
            scalar_values[valid_mask] = weights[valid_mask] @ vertex_scalars
        return scalar_values

    def metsim_triangle_scalar_values(
        self,
        matrix,
        *,
        color_order="RBG",
        x_component=2,
        y_component=1,
        scale_factor=1.0,
        channel_scalar_map=None,
        return_details=False,
    ):
        matrix_arr = np.asarray(matrix, dtype=float)
        if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
            raise ValueError("matrix must be a square 2D array.")
        x_component = max(1, int(x_component))
        y_component = max(1, int(y_component))
        max_component = max(x_component, y_component)
        gradients = np.column_stack(
            [
                self.dimreduce_matrix(
                    matrix_arr,
                    method="diffusion",
                    output_dim=component_index,
                    scale_factor=scale_factor,
                    norm=False,
                )
                for component_index in range(1, max_component + 1)
            ]
        )
        x_values = np.asarray(gradients[:, x_component - 1], dtype=float)
        y_values = np.asarray(gradients[:, y_component - 1], dtype=float)
        model = self.triangular_rgb_model(x_values, y_values, color_order=color_order)
        scalar_values = self.triangular_rgb_scalar_from_model(
            x_values,
            y_values,
            model,
            channel_scalar_map=channel_scalar_map,
        )
        if not return_details:
            return scalar_values
        return {
            "values": np.asarray(scalar_values, dtype=float),
            "gradient_x": np.asarray(x_values, dtype=float),
            "gradient_y": np.asarray(y_values, dtype=float),
            "weights": np.asarray(self.triangular_rgb_weights_from_model(x_values, y_values, model), dtype=float),
            "rgb_colors": np.asarray(self.triangular_rgb_colors_from_model(x_values, y_values, model), dtype=float),
            "model": {
                "vertices": np.asarray(model["vertices"], dtype=float),
                "anchor_points": np.asarray(model["anchor_points"], dtype=float),
                "vertex_colors": np.asarray(model["vertex_colors"], dtype=float),
                "order": str(model.get("order", color_order)),
                "fit_mode": "triangle",
            },
        }

    def align_gradients_procrustes(
        self,
        gradients,
        reference=None,
        n_iter=10,
        tol=1e-5,
        return_reference=False,
    ):
        """
        Align gradient arrays using BrainSpace generalized Procrustes analysis.

        Parameters:
            gradients: list of arrays or single array, shape (n_nodes,) or (n_nodes, n_components)
            reference: optional reference gradient to initialize alignment
        """
        try:
            from brainspace.gradient.alignment import procrustes_alignment
        except Exception as exc:
            raise ImportError(
                "brainspace is required for Procrustes alignment. Install it via `pip install brainspace`."
            ) from exc

        if isinstance(gradients, np.ndarray):
            gradient_list = [gradients]
        else:
            gradient_list = list(gradients)

        prepared = []
        for grad in gradient_list:
            arr = np.asarray(grad, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            prepared.append(arr)

        ref_arr = None
        if reference is not None:
            ref_arr = np.asarray(reference, dtype=float)
            if ref_arr.ndim == 1:
                ref_arr = ref_arr[:, None]

        aligned = procrustes_alignment(
            prepared,
            reference=ref_arr,
            n_iter=n_iter,
            tol=tol,
            return_reference=return_reference,
        )

        if return_reference:
            aligned_list, ref_out = aligned
        else:
            aligned_list = aligned
            ref_out = None

        squeezed = []
        for orig, aln in zip(prepared, aligned_list):
            if orig.shape[1] == 1:
                squeezed.append(aln[:, 0])
            else:
                squeezed.append(aln)

        if return_reference:
            if ref_out is not None and ref_out.ndim == 2 and ref_out.shape[1] == 1:
                ref_out = ref_out[:, 0]
            return squeezed, ref_out
        return squeezed


    @staticmethod
    def free_energy(gradient_values, adj_dict=None, labels=None, bins=None, mode="all_pairs"):
        """
        Compute scalar energy/entropy for one gradient vector.

        Parameters
        ----------
        gradient_values : array-like or dict
            Gradient values per node.
        adj_dict : dict, optional
            Adjacency dictionary `{node: neighbors}`. Used when `mode='adjacency'`.
        labels : array-like, optional
            Node labels matching `gradient_values` order.
        bins : int, optional
            Number of bins used for entropy estimation. If None, uses
            `round(2 * n_nodes**(1/3))` with `n_nodes = len(gradient_values)`.
        mode : {'all_pairs', 'adjacency'}, optional
            `all_pairs`: sum over all unique node pairs.
            `adjacency`: sum only over undirected adjacency edges from `adj_dict`.

        Returns
        -------
        energy : float
            Sum of squared gradient differences.
        entropy : float
            Shannon entropy (base 2) of gradient values.
        """

        if isinstance(gradient_values, dict):
            if labels is None:
                labels = list(gradient_values.keys())
            vals = np.array([gradient_values[label] for label in labels], dtype=float)
        else:
            vals = np.asarray(gradient_values, dtype=float).reshape(-1)
            if labels is None:
                labels = list(range(vals.shape[0]))

        labels = [NetTools._normalize_label(lbl) for lbl in labels]
        if vals.shape[0] != len(labels):
            raise ValueError("gradient_values and labels must have the same length.")

        finite_mask = np.isfinite(vals)
        if not np.any(finite_mask):
            return float("nan"), float("nan")

        vals_finite = vals[finite_mask]
        labels_finite = [labels[i] for i in np.where(finite_mask)[0]]
        n = vals_finite.shape[0]

        if n <= 1:
            energy = 0.0
        elif str(mode).lower() == "adjacency" and adj_dict is not None:
            label_to_idx = {label: i for i, label in enumerate(labels_finite)}
            seen_edges = set()
            energy = 0.0
            for node, neighbors in adj_dict.items():
                node_key = NetTools._normalize_label(node)
                i = label_to_idx.get(node_key)
                if i is None:
                    continue
                for neighbor in neighbors:
                    nb_key = NetTools._normalize_label(neighbor)
                    j = label_to_idx.get(nb_key)
                    if j is None or i == j:
                        continue
                    edge = (i, j) if i < j else (j, i)
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    diff = vals_finite[i] - vals_finite[j]
                    energy += float(diff * diff)
        else:
            # Sum_{i<j} (x_i - x_j)^2 = n * sum(x^2) - (sum x)^2
            sum_sq = float(np.sum(vals_finite * vals_finite))
            sum_x = float(np.sum(vals_finite))
            energy = float(n * sum_sq - sum_x * sum_x)

        vmin = float(np.nanmin(vals_finite))
        vmax = float(np.nanmax(vals_finite))
        if bins is None:
            n_nodes = int(vals.shape[0])
            bins_eff = int(round(2 * (n_nodes ** (1.0 / 3.0))))
        else:
            bins_eff = int(bins)
        bins_eff = int(max(2, bins_eff))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            entropy = 0.0
        else:
            counts, _ = np.histogram(vals_finite, bins=bins_eff, range=(vmin, vmax))
            probs = counts.astype(float)
            probs = probs[probs > 0]
            probs = probs / np.sum(probs)
            entropy = float(-np.sum(probs * np.log10(probs)))

        return energy, entropy
