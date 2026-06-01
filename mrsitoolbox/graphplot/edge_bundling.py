from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

__all__ = [
    "BundledEdges2D",
    "bundle_edges_2d",
    "direct_edges_2d",
    "hammer_bundle_edges_2d",
    "polylines_to_segments",
    "split_bundled_dataframe_2d",
]


@dataclass
class BundledEdges2D:
    """Container for 2D bundled edge geometry with fixed node positions."""

    points: np.ndarray
    edge_pairs: np.ndarray
    weights: np.ndarray
    polylines: list[np.ndarray]
    method: str
    bundled_dataframe: object | None = None


def _coerce_points(points) -> np.ndarray:
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`points` must have shape (N, 2).")
    if not np.all(np.isfinite(coords)):
        raise ValueError("`points` contains non-finite values.")
    return coords


def _coerce_edge_pairs(edge_pairs, n_points: int) -> np.ndarray:
    pairs = np.asarray(edge_pairs, dtype=int)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("`edge_pairs` must have shape (M, 2).")
    valid = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 0] < int(n_points))
        & (pairs[:, 1] < int(n_points))
        & (pairs[:, 0] != pairs[:, 1])
    )
    pairs = pairs[valid]
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(pairs, dtype=int)


def _coerce_weights(weights, n_edges: int) -> np.ndarray:
    if n_edges <= 0:
        return np.zeros(0, dtype=float)
    if weights is None:
        return np.ones(n_edges, dtype=float)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.shape[0] != n_edges:
        raise ValueError("`weights` must be None or have the same length as `edge_pairs`.")
    values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=1.0)
    values[values <= 0.0] = 1.0
    return values


def _split_nan_separated_polylines(values: np.ndarray) -> list[np.ndarray]:
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected a 2D array with shape (K, 2).")
    polylines = []
    start = 0
    breaks = np.flatnonzero(~np.all(np.isfinite(values), axis=1))
    for stop in list(breaks.tolist()) + [values.shape[0]]:
        chunk = np.asarray(values[start:stop], dtype=float)
        chunk = chunk[np.all(np.isfinite(chunk), axis=1)]
        if chunk.shape[0] >= 2:
            polylines.append(chunk)
        start = stop + 1
    return polylines


def split_bundled_dataframe_2d(path_dataframe) -> list[np.ndarray]:
    """Convert a datashader bundled edge DataFrame into 2D polylines."""

    if path_dataframe is None:
        return []
    try:
        values = np.asarray(path_dataframe[["x", "y"]].to_numpy(dtype=float), dtype=float)
    except Exception as exc:
        raise ValueError("Bundled edge DataFrame must expose numeric 'x' and 'y' columns.") from exc
    if values.size == 0:
        return []
    return _split_nan_separated_polylines(values)


def polylines_to_segments(polylines) -> np.ndarray:
    """Convert a list of 2D polylines into Matplotlib LineCollection segments."""

    segments = []
    for polyline in list(polylines or []):
        points = np.asarray(polyline, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            continue
        if not np.all(np.isfinite(points)):
            continue
        for idx in range(points.shape[0] - 1):
            segments.append(points[idx : idx + 2, :])
    if not segments:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(segments, dtype=float)


def direct_edges_2d(points, edge_pairs, *, weights=None) -> BundledEdges2D:
    """Return straight 2D edge polylines without moving node positions."""

    coords = _coerce_points(points)
    pairs = _coerce_edge_pairs(edge_pairs, coords.shape[0])
    edge_weights = _coerce_weights(weights, pairs.shape[0])
    polylines = [np.asarray(coords[pair, :], dtype=float) for pair in pairs.tolist()]
    return BundledEdges2D(
        points=coords,
        edge_pairs=pairs,
        weights=edge_weights,
        polylines=polylines,
        method="direct",
        bundled_dataframe=None,
    )


def _build_datashader_inputs(points, edge_pairs, weights):
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required for hammer edge bundling.") from exc

    point_df = pd.DataFrame(np.asarray(points, dtype=float), columns=["x", "y"])
    edge_df = pd.DataFrame(
        {
            "source": np.asarray(edge_pairs[:, 0], dtype=int),
            "target": np.asarray(edge_pairs[:, 1], dtype=int),
            "weight": np.asarray(weights, dtype=float),
        }
    )
    return point_df, edge_df


def hammer_bundle_edges_2d(
    points,
    edge_pairs,
    *,
    weights=None,
    use_dask=False,
    iterations=6,
    accuracy=500,
) -> BundledEdges2D:
    """Bundle fixed 2D edges with datashader's hammer bundler."""

    coords = _coerce_points(points)
    pairs = _coerce_edge_pairs(edge_pairs, coords.shape[0])
    edge_weights = _coerce_weights(weights, pairs.shape[0])
    if pairs.shape[0] == 0:
        return BundledEdges2D(
            points=coords,
            edge_pairs=pairs,
            weights=edge_weights,
            polylines=[],
            method="hammer",
            bundled_dataframe=None,
        )

    try:
        os.environ.setdefault(
            "NUMBA_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "numba_cache"),
        )
        import datashader.bundling as bundling
    except Exception as exc:
        raise ImportError(
            "datashader.bundling is required for hammer edge bundling."
        ) from exc

    point_df, edge_df = _build_datashader_inputs(coords, pairs, edge_weights)
    bundled_df = bundling.hammer_bundle(
        point_df,
        edge_df,
        weight="weight",
        use_dask=bool(use_dask),
        iterations=int(iterations),
        accuracy=int(accuracy),
    )
    polylines = split_bundled_dataframe_2d(bundled_df)
    return BundledEdges2D(
        points=coords,
        edge_pairs=pairs,
        weights=edge_weights,
        polylines=polylines,
        method="hammer",
        bundled_dataframe=bundled_df,
    )


def bundle_edges_2d(
    points,
    edge_pairs,
    *,
    weights=None,
    method="hammer",
    use_dask=False,
    iterations=6,
    accuracy=500,
) -> BundledEdges2D:
    """Bundle or directly connect fixed 2D edges without moving node positions."""

    mode = str(method or "hammer").strip().lower()
    if mode in {"direct", "straight", "none"}:
        return direct_edges_2d(points, edge_pairs, weights=weights)
    if mode in {"hammer", "bundled", "bundle"}:
        return hammer_bundle_edges_2d(
            points,
            edge_pairs,
            weights=weights,
            use_dask=use_dask,
            iterations=iterations,
            accuracy=accuracy,
        )
    raise ValueError("`method` must be one of: 'hammer', 'bundled', 'direct', 'straight', 'none'.")
