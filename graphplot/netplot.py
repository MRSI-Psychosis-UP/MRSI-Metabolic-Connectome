import numpy as np
from scipy.interpolate import CubicSpline
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
from PIL import Image
from os.path  import join, split
from dipy.viz import actor


class NetPlot:
    def __init__(self,window):
        self.scene  = window.Scene()
        self.mni_template = datasets.load_mni152_template()
        pass



    def draw_geometric_curve(self,labels_points,centroid_dict, num_curve_points = 50000,start_label_idx=0):
        """
        Draws a geometric curve using cubic splines through the centroids of specified brain regions,
        and returns a binary mask of the curve.

        Parameters:
        - mni_template (np.ndarray): 3D numpy array representing the MNI template (used only for shape).
        - label_image (np.ndarray): 3D numpy array with integer labels assigned to brain regions.
        - labels (list of int): List of integer labels for brain regions to pass through with the curve.

        Returns:
        - np.ndarray: Binary mask of the curve with the same shape as the input images.
        """
        # Calculate centroids of the regions specified by the labels
        _centroids = np.array([centroid_dict[label] for label in labels_points])
        
        # Extract x, y, z coordinates for cubic spline
        N = len(_centroids)
        t = np.arange(N)
        x = _centroids[:, 0]
        y = _centroids[:, 1]
        z = _centroids[:, 2]
        spline_x = CubicSpline(t, x)
        spline_y = CubicSpline(t, y)
        spline_z = CubicSpline(t, z)
        # Generate parameter t values
        # Generate t values for the curve, excluding the segments corresponding to the first k labels
        if start_label_idx == 0:
            # No segments are deleted; use the full range
            t_new = np.linspace(t[0], t[-1], num_curve_points)
        else:
            # Exclude segments between the first k centroids
            # Create a t_new array that skips the intervals [t[i], t[i+1]] for i in range(k - 1)
            # We'll build t_new in segments starting from t[k - 1]
            t_new = np.linspace(t[start_label_idx], t[-1], num_curve_points)

        # Evaluate the splines at the new t values
        x_new = spline_x(t_new)
        y_new = spline_y(t_new)
        z_new = spline_z(t_new)

        # Combine the evaluated points into a single array of shape (num_curve_points, 3)
        curve_points = np.vstack([x_new, y_new, z_new]).T

        return None, curve_points

    def add_fibre(self,label_points,centroid_dict,homotopy_image,label_image,start_label=None,
                  num_curve_points=1000,linewidth=10,opacity=0.5,colormap="plasma",uniform_colors=False):
        """
        - Constructs and adds a metabolic fibre to scene.
        """
        if start_label is not None:
            k = label_points.index(start_label)
        else:k=0
        _, fibre_coord = self.draw_geometric_curve(label_points, centroid_dict,
                                                        num_curve_points=num_curve_points,
                                                        start_label_idx=k)

        
        colors_fibre,homotopy_values_curve = self.get_fibre_colors(label_points[k::], homotopy_image, label_image,
                                                        colormap, fibre_coord.shape[0],
                                                        uniform=uniform_colors)
        # curve_actor = actor.line([fibre_coord], colors=colors_main_fibre_LH, linewidth=linewidth,fake_tube=True)
        curve_actor = actor.streamtube([fibre_coord], colors=colors_fibre, linewidth=linewidth,opacity=opacity)
        self.scene.add(curve_actor)
        return homotopy_values_curve


    def add_brain(self, mni_template, hemisphere="lh", label_image=None, parcel_labels_list=None,opacity=0.23):
        # Load the brain data
        brain_data = mni_template.get_fdata()

        # Create a mask based on the selected hemisphere
        mask = np.ones(brain_data.shape, dtype=bool)
        if hemisphere is not None:
            if hemisphere == "lh":
                # Mask out the right hemisphere (X > 0)
                mask[brain_data.shape[0] // 2:, :, :] = False
            elif hemisphere == "rh":
                # Mask out the left hemisphere (X <= 0)
                mask[:brain_data.shape[0] // 2, :, :] = False

        # If label_image and parcel_labels_list are provided, apply the parcel mask
        if label_image is not None and parcel_labels_list is not None:
            # Create a mask for the specified parcels
            parcel_mask = np.isin(label_image, parcel_labels_list)
            # Combine the parcel mask with the hemisphere mask
            mask = mask & parcel_mask

        # Apply the final mask to the brain data
        masked_brain_data = brain_data * mask

        # Create the actor for the masked brain data
        brain_actor = actor.contour_from_roi(masked_brain_data, color=(1, 1, 1), opacity=opacity)
        self.scene.add(brain_actor)

    def __add_brain(self,mni_template,hemisphere="lh",label_image=None,parcel_labels_list=None):
        # Add the transparent brain and curve as before
        brain_data    = mni_template.get_fdata()
        # Create a mask based on the selected hemisphere
        mask = np.ones(brain_data.shape, dtype=bool)
        if hemisphere == "lh":
            # Mask out the right hemisphere (X > 0)
            mask[brain_data.shape[0] // 2:, :, :] = False
        elif hemisphere == "rh":
            # Mask out the left hemisphere (X <= 0)
            mask[:brain_data.shape[0] // 2, :, :] = False
        # Apply the mask to the brain data
        masked_brain_data = brain_data * mask
        # Create the scene and add the modified brain
        brain_actor = actor.contour_from_roi(masked_brain_data, color=(1, 1, 1), opacity=0.23)  # Transparent brain
        self.scene.add(brain_actor)

    def plot_rich_club_nodes(self,connectivity_matrix, centroids, rich_club_nodes_indices,homotopy_values_rc,colormap,marker_scale=1,rc_labels=None):
        """
        Plots the rich club nodes on a 3D brain using nilearn.view_markers, with marker sizes proportional to their degree.
        
        :param binarized_matrix: 2D numpy array representing the binarized adjacency matrix.
        :param coordinates: Numpy array with shape (n_nodes, 3) representing the 3D coordinates of each node.
        :param rich_club_nodes_indices: List of node indices corresponding to rich club nodes.
        """
        # Compute degrees of all nodes
        degrees = np.sum(connectivity_matrix, axis=1)
        # Get the coordinates and degrees of the rich club nodes
        rich_club_coords  = centroids[rich_club_nodes_indices]
        rich_club_degrees = degrees[rich_club_nodes_indices]
        # Normalize degree sizes for better visualization (optional: scale factor)
        marker_sizes = marker_scale* rich_club_degrees**3
        norm = plt.Normalize(vmin=np.min(homotopy_values_rc), vmax=np.max(homotopy_values_rc))
        colors = colormap(norm(homotopy_values_rc))
        # Plot rich club nodes in 3D on brain template with marker sizes based on degree
        # view = plotting.view_markers(rich_club_coords, marker_size=marker_sizes)
        view = plotting.view_markers(rich_club_coords, marker_size=marker_sizes, marker_color=colors,marker_labels=rc_labels)
        # Open the view in a browser
        view.open_in_browser()
        return view  # Return view object for further customization if needed

    @staticmethod
    def has_duplicate_pair(existing_triplets, new_triplet):
        """
        Checks if any pair in the new_triplet is already present in the existing_triplets.
        """
        # Set to track all pairs in existing triplets
        seen_pairs = set()
        # Populate seen_pairs with pairs from existing triplets
        for triplet in existing_triplets:
            pairs = {(triplet[0], triplet[1]), (triplet[0], triplet[2]), (triplet[1], triplet[2])}
            seen_pairs.update(pairs)
        # Generate pairs for the new triplet and check if any are in seen_pairs
        new_triplet_pairs = {(new_triplet[0], new_triplet[1]), 
                            (new_triplet[0], new_triplet[2]), 
                            (new_triplet[1], new_triplet[2])}
        # If any pair in new_triplet is in seen_pairs, return 1 (indicating a duplicate), else 0
        if new_triplet_pairs & seen_pairs:
            return 1
        else:
            return 0

    @staticmethod
    def interpolate_array(original_array, M):
        """
        Interpolates an array to a new array of length M while preserving the original values.

        Parameters:
            original_array (numpy.ndarray): Input array of shape (N,) with values between 0 and 1.
            M (int): Desired length of the output array.

        Returns:
            numpy.ndarray: Interpolated array of length M.
        """
        if len(original_array) > M:
            raise ValueError("M must be greater than or equal to the length of the original array.")
        
        N = len(original_array)
        # Compute the indices of the original points in the new interpolated array
        original_indices = np.linspace(0, M - 1, N)
        # Create the full range of indices for the interpolated array
        new_indices = np.arange(M)
        # Interpolate using numpy's interpolation function
        interpolated_array = np.interp(new_indices, original_indices, original_array)
        
        return interpolated_array


    def get_fibre_colors(self, label_endpoints, homotopy_image, label_image, color_map, M=1000, uniform=False):
        homotopy_values_curve = np.zeros(len(label_endpoints))
        for i, node_label in enumerate(label_endpoints):
            # Get a mask of where `label_image` equals `node_label`
            mask = label_image == node_label
            # Extract homotopy values using the mask
            homotopy_values = homotopy_image[mask]
            # Filter out zero values and compute the mean
            nonzero_homotopy_values = homotopy_values[homotopy_values != 0]
            if nonzero_homotopy_values.size > 0:
                homotopy_values_curve[i] = np.mean(nonzero_homotopy_values)
            else:
                homotopy_values_curve[i] = 0 

        norm = plt.Normalize(vmin=np.min(homotopy_image), vmax=np.max(homotopy_image))
        homotopy_curve_norm = norm(homotopy_values_curve).data
        homotopy_curve_norm_inter = list()
        K = int(np.round(M / (len(homotopy_curve_norm) - 1)))
        for i in range(len(homotopy_curve_norm) - 1):
            h_value_i = homotopy_curve_norm[i]
            h_value_j = homotopy_curve_norm[i + 1]
            _inter_values = np.linspace(h_value_i, h_value_j, K)
            homotopy_curve_norm_inter.extend(_inter_values)
        homotopy_curve_norm_inter = np.array(homotopy_curve_norm_inter)
        k = M - homotopy_curve_norm_inter.shape[0]
        if k > 0:
            last_element = homotopy_curve_norm_inter[-1]
            extra_elements = np.full((k,), last_element)  # Fixed to match dimensionality
            homotopy_curve_norm_inter = np.concatenate([homotopy_curve_norm_inter, extra_elements])
        homotopy_curve_norm_inter = homotopy_curve_norm_inter[0:M]
        if uniform:
            colors = color_map(np.linspace(min(homotopy_curve_norm), max(homotopy_curve_norm), M))[:, :3]  # Colormap for each segment of the curve
        else:    
            colors = color_map(homotopy_curve_norm_inter)[:, :3]
        return colors, homotopy_values_curve

    def snapshot(self,brain_data,window,scene, view="top", filepath="test"):
        """
        Capture a snapshot of the scene from a specific anatomical viewpoint.

        Parameters:
        - scene: The DIPY scene to capture.
        - view: String defining the viewpoint ('top', 'left', 'right', 'front', 'back').
        - filename: Name of the file to save the snapshot.

        Returns:
        - None
        """
        OUTDIRPATH = "."  # Adjust to your output directory if needed
        brain_center = self.compute_brain_center(brain_data)
        print(f"Computed brain center: {brain_center}")
        # Define camera positions for anatomical views
        camera_positions = {
            "top": (brain_center[0], brain_center[1], brain_center[2] + 500),
            "left": (brain_center[0] - 500, brain_center[1], brain_center[2]),
            "right": (brain_center[0] + 500, brain_center[1], brain_center[2]),
            "front": (brain_center[0], brain_center[1] + 500, brain_center[2]),
            "back": (brain_center[0], brain_center[1] - 500, brain_center[2])
        }
        # Get the desired camera position
        position = camera_positions.get(view.lower())
        if position is None:
            raise ValueError(f"Unknown view '{view}'. Choose from: {list(camera_positions.keys())}")
        # Set the focal point (center of the object/scene)
        scene.GetActiveCamera().SetFocalPoint(0, 0, 0)  # Adjust if your object isn't centered at (0, 0, 0)
        # Set the camera position
        scene.GetActiveCamera().SetPosition(*position)
        # Set the view-up direction (Z-axis as "up")
        scene.GetActiveCamera().SetViewUp(0, 0, 1)
        # Capture the snapshot
        snapshot = window.snapshot(scene, size=(800, 800), offscreen=True)
        # Save the snapshot to the specified file
        output_file = join(OUTDIRPATH, filepath)
        Image.fromarray(snapshot).save(output_file)
        print(f"Snapshot saved as {output_file}")

    @staticmethod
    def compute_brain_center(brain_data):
        """
        Compute the center of the brain data to use as the camera focal point.

        Parameters:
        - brain_data: 3D numpy array or similar data representing the brain.

        Returns:
        - Tuple (x, y, z) representing the center.
        """
        coords = np.array(np.nonzero(brain_data))  # Get non-zero voxel coordinates
        center = coords.mean(axis=1)  # Compute the mean for each axis
        return tuple(center)
    


    def add_gm_adjacency(self, adjacency_mat, centroids,
                        node_radius=3, node_color=(0.8, 0.1, 0.1),
                        edge_color=(0.1, 0.1, 0.8), edge_opacity=0.4,
                        edge_threshold=None, node_labels=None):
        """
        Plot GM adjacency graph (nodes + edges) onto the existing MNI brain scene.

        Parameters
        ----------
        adjacency_mat : (N, N) np.ndarray
            Adjacency matrix of GM network (binary or weighted).
        centroids : (N, 3) np.ndarray
            MNI/world coordinates of each node centroid.
        node_radius : float
            Radius of the spheres for nodes.
        node_color : tuple or array
            RGB color of the nodes (0-1 range) or array of colors.
        edge_color : tuple
            RGB color of the edges.
        edge_opacity : float
            Opacity of the edges (0-1).
        edge_threshold : float | None
            If set, only plot edges with weight > threshold.
        node_labels : list of str | None
            Optional labels for each node, same length as centroids.
        """
        N = adjacency_mat.shape[0]

        # --- Nodes ---
        sphere_actor = actor.sphere(centroids, colors=node_color,
                                    radii=node_radius * np.ones(N))
        self.scene.add(sphere_actor)

        # --- Node labels ---
        if node_labels is not None:
            for label, pos in zip(node_labels, centroids):
                text_actor = actor.text_3d(str(label),
                                        position=tuple(pos),
                                        color=(0, 0, 0),
                                        font_size=3,
                                        justification="center")
                self.scene.add(text_actor)

        # --- Edges ---
        edge_coords, edge_colors = [], []
        for i in range(N):
            for j in range(i + 1, N):
                w = adjacency_mat[i, j]
                if w == 0:
                    continue
                if edge_threshold is not None and w < edge_threshold:
                    continue
                edge_coords.append([centroids[i], centroids[j]])
                edge_colors.append(edge_color)

        if edge_coords:
            line_actor = actor.line(edge_coords, colors=edge_colors, opacity=edge_opacity)
            self.scene.add(line_actor)

        return self.scene
