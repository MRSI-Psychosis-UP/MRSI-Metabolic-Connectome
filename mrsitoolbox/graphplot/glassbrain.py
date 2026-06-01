"""PyVista glass-brain rendering helpers."""

from __future__ import annotations

import os

import numpy as np

try:
    import pyvista as pv
except Exception as exc:  # pragma: no cover - depends on optional runtime package
    pv = None
    _PYVISTA_IMPORT_ERROR = exc
else:
    _PYVISTA_IMPORT_ERROR = None

try:
    from matplotlib import colors as mcolors
    from nilearn import datasets, surface
except Exception as exc:  # pragma: no cover - import errors are surfaced lazily
    datasets = None
    surface = None
    mcolors = None
    _SURFACE_IMPORT_ERROR = exc
else:
    _SURFACE_IMPORT_ERROR = None


class GlassBrainPlotter:
    """Render parcel nodes and paths on a transparent fsaverage brain mesh."""

    _brain_mesh_cache = None
    _runtime_ready = False

    def __init__(
        self,
        *,
        brain_opacity=0.14,
        background="white",
        window_size=(1200, 900),
        path_radius_scale=0.08,
        off_screen=False,
    ):
        self.brain_opacity = float(brain_opacity)
        self.background = background
        self.window_size = tuple(int(value) for value in window_size)
        self.path_radius_scale = float(path_radius_scale)
        self.off_screen = bool(off_screen)
        self.plotter = None

    @classmethod
    def require_pyvista(cls):
        if pv is None:
            raise ImportError("PyVista is required for glass-brain rendering.") from _PYVISTA_IMPORT_ERROR
        if datasets is None or surface is None:
            raise ImportError("nilearn surface utilities are required for glass-brain rendering.") from _SURFACE_IMPORT_ERROR
        if cls._runtime_ready:
            return
        try:
            pv.OFF_SCREEN = False
        except Exception:
            pass
        if os.environ.get("PYVISTA_USE_XVFB", "").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                pv.start_xvfb()
            except Exception:
                pass
        cls._runtime_ready = True

    @classmethod
    def load_brain_mesh(cls):
        if cls._brain_mesh_cache is not None:
            return cls._brain_mesh_cache
        cls.require_pyvista()
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        coords_l, faces_l = surface.load_surf_mesh(fsaverage.pial_left)
        coords_r, faces_r = surface.load_surf_mesh(fsaverage.pial_right)
        coords = np.vstack([coords_l, coords_r]).astype(float)
        faces_r = np.asarray(faces_r, dtype=np.int64) + int(coords_l.shape[0])
        faces = np.vstack([np.asarray(faces_l, dtype=np.int64), faces_r])
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        cls._brain_mesh_cache = pv.PolyData(coords, faces_pv)
        return cls._brain_mesh_cache

    @staticmethod
    def _as_rgba_array(colors, n_items, default=(0.6, 0.6, 0.6, 0.25)):
        if mcolors is None:
            base = np.asarray(default, dtype=float).reshape(4)
            return np.tile(base.reshape(1, 4), (int(n_items), 1))
        if colors is None:
            base = np.asarray(default, dtype=float).reshape(4)
            return np.tile(base.reshape(1, 4), (int(n_items), 1))
        if isinstance(colors, str):
            base = np.asarray(mcolors.to_rgba(colors), dtype=float)
            return np.tile(base.reshape(1, 4), (int(n_items), 1))
        arr = np.asarray(colors, dtype=object)
        if arr.ndim == 1 and arr.size in (3, 4):
            base = np.asarray(mcolors.to_rgba(arr.astype(float).tolist()), dtype=float)
            return np.tile(base.reshape(1, 4), (int(n_items), 1))
        out = []
        for item in arr.tolist():
            out.append(mcolors.to_rgba(item))
        if len(out) != int(n_items):
            raise ValueError("Color count must match the number of rendered items.")
        return np.asarray(out, dtype=float)

    @staticmethod
    def _as_size_array(sizes, n_items, default=2.4):
        if sizes is None:
            return np.full(int(n_items), float(default), dtype=float)
        if np.isscalar(sizes):
            return np.full(int(n_items), float(sizes), dtype=float)
        arr = np.asarray(sizes, dtype=float).reshape(-1)
        if arr.shape[0] != int(n_items):
            raise ValueError("Size count must match the number of rendered items.")
        return arr

    @staticmethod
    def _polyline_mesh(path_coords):
        coords = np.asarray(path_coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 2:
            return None
        if not np.all(np.isfinite(coords)):
            return None
        mesh = pv.PolyData(coords)
        mesh.lines = np.hstack([[coords.shape[0]], np.arange(coords.shape[0], dtype=np.int64)])
        return mesh

    def _brain_mesh_for_hemisphere(self, hemisphere_mode):
        mesh = self.load_brain_mesh()
        hemi = str(hemisphere_mode or "both").strip().lower()
        if hemi not in {"lh", "rh"}:
            return mesh
        points = np.asarray(mesh.points, dtype=float)
        keep_points = points[:, 0] < 0 if hemi == "lh" else points[:, 0] > 0
        try:
            return mesh.extract_points(keep_points, adjacent_cells=True)
        except Exception:
            return mesh

    def build_plotter(
        self,
        *,
        node_coords,
        paths=None,
        path_colors=None,
        path_widths=None,
        node_colors=None,
        node_sizes=None,
        anchor_coords=None,
        anchor_colors=None,
        anchor_labels=None,
        anchor_size=4.0,
        hemisphere_mode="both",
        title=None,
    ):
        self.require_pyvista()
        coords = np.asarray(node_coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("node_coords must have shape (N, 3).")
        finite_nodes = np.all(np.isfinite(coords), axis=1)
        plotter = pv.Plotter(off_screen=bool(self.off_screen), window_size=self.window_size)
        plotter.set_background(self.background)
        plotter.add_mesh(
            self._brain_mesh_for_hemisphere(hemisphere_mode),
            color="lightgray",
            opacity=float(np.clip(self.brain_opacity, 0.0, 1.0)),
            smooth_shading=True,
        )

        rgba_nodes = self._as_rgba_array(node_colors, coords.shape[0])
        radii = self._as_size_array(node_sizes, coords.shape[0], default=2.4)
        for xyz, rgba, radius, is_finite in zip(coords, rgba_nodes, radii, finite_nodes):
            if not is_finite or float(radius) <= 0.0 or float(rgba[3]) <= 0.0:
                continue
            sphere = pv.Sphere(radius=float(radius), center=np.asarray(xyz, dtype=float))
            plotter.add_mesh(
                sphere,
                color=tuple(np.asarray(rgba[:3], dtype=float).tolist()),
                opacity=float(np.clip(float(rgba[3]), 0.0, 1.0)),
                smooth_shading=True,
            )

        path_arrays = [] if paths is None else [np.asarray(path, dtype=float) for path in list(paths)]
        rgba_paths = self._as_rgba_array(path_colors, len(path_arrays), default=(0.1, 0.1, 0.1, 1.0))
        widths = self._as_size_array(path_widths, len(path_arrays), default=2.5)
        for path_coords, rgba, width in zip(path_arrays, rgba_paths, widths):
            mesh = self._polyline_mesh(path_coords)
            if mesh is None:
                continue
            radius = max(0.08, float(width) * float(self.path_radius_scale))
            tube = mesh.tube(radius=radius)
            plotter.add_mesh(
                tube,
                color=tuple(np.asarray(rgba[:3], dtype=float).tolist()),
                opacity=float(np.clip(float(rgba[3]), 0.0, 1.0)),
                smooth_shading=True,
            )

        anchors = np.asarray(anchor_coords, dtype=float) if anchor_coords is not None else np.zeros((0, 3), dtype=float)
        if anchors.ndim == 2 and anchors.shape[1] == 3 and anchors.shape[0] > 0:
            rgba_anchors = self._as_rgba_array(anchor_colors, anchors.shape[0], default=(0.0, 0.0, 0.0, 1.0))
            for xyz, rgba in zip(anchors, rgba_anchors):
                if not np.all(np.isfinite(xyz)):
                    continue
                sphere = pv.Sphere(radius=float(anchor_size), center=np.asarray(xyz, dtype=float))
                plotter.add_mesh(
                    sphere,
                    color=tuple(np.asarray(rgba[:3], dtype=float).tolist()),
                    opacity=float(np.clip(float(rgba[3]), 0.0, 1.0)),
                    smooth_shading=True,
                )
            label_source = [] if anchor_labels is None else list(anchor_labels)
            labels = [str(label) for label in label_source]
            if labels and len(labels) == anchors.shape[0]:
                try:
                    plotter.add_point_labels(anchors, labels, font_size=12, text_color="black", point_size=0)
                except Exception:
                    pass

        if title:
            try:
                plotter.add_text(str(title), font_size=12, color="black")
            except Exception:
                pass
        try:
            plotter.view_xy()
            plotter.camera.zoom(1.12)
        except Exception:
            pass
        self.plotter = plotter
        return plotter

    def show_paths(self, **kwargs):
        plotter = self.build_plotter(**kwargs)
        plotter.show(title=str(kwargs.get("title") or "Glass Brain Paths"))
        return plotter

    def screenshot(self, camera_view="dorsal", **kwargs):
        previous_off_screen = self.off_screen
        self.off_screen = True
        try:
            plotter = self.build_plotter(**kwargs)
        finally:
            self.off_screen = previous_off_screen
        view = str(camera_view or "dorsal").strip().lower()
        if view == "dorsal":
            plotter.view_xy()
        elif view == "coronal":
            plotter.view_xz()
        elif view == "lateral":
            plotter.view_yz()
        image = plotter.screenshot(return_img=True)
        plotter.close()
        return image

    def three_view_screenshots(self, views=("dorsal", "coronal", "lateral"), **kwargs):
        return {
            str(view): self.screenshot(camera_view=str(view), **kwargs)
            for view in tuple(views)
        }

    def save_three_view_panel(
        self,
        output_path,
        *,
        views=("dorsal", "coronal", "lateral"),
        show_titles=False,
        dpi=200,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        images = self.three_view_screenshots(views=views, **kwargs)
        view_names = [str(view) for view in tuple(views)]
        fig, axes = plt.subplots(
            1,
            len(view_names),
            figsize=(4.2 * len(view_names), 4.0),
            facecolor="white",
            constrained_layout=True,
        )
        if len(view_names) == 1:
            axes = [axes]
        for ax, view_name in zip(axes, view_names):
            ax.imshow(images[view_name])
            ax.set_axis_off()
            if show_titles:
                ax.set_title(view_name.capitalize(), fontsize=11)
        fig.savefig(str(output_path), dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return str(output_path)


__all__ = ["GlassBrainPlotter"]
