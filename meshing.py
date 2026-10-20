"""Watertight triangle meshes for height fields over a pixel grid.

Each selected pixel becomes a square column from z=0 up to its top surface.
Heights live on pixel corners, so neighbouring columns share edges and the
top surface stays continuous across pixels and across separate parts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (n, 3) float32, millimetres
    faces: np.ndarray  # (m, 3) int32, counter-clockwise seen from outside
    corners: np.ndarray  # (n,) flat index of each vertex in the (h+1, w+1) corner grid


def without_pinches(regions: np.ndarray, max_passes: int = 100) -> np.ndarray:
    """Two pixels of one region touching only at a corner would share a
    single vertical edge between four walls, which slicers report as a
    non-manifold mesh. Each pinch is closed by giving the region one of the
    two pixels across it. Returns the fixed region map; raises if it does
    not settle, which would mean the rule can cycle."""
    regions = regions.copy()
    for _ in range(max_passes):
        tl, tr, bl, br = regions[:-1, :-1], regions[:-1, 1:], regions[1:, :-1], regions[1:, 1:]
        main = (tl == br) & (tl != tr) & (tl != bl)
        anti = (tr == bl) & (tr != tl) & (tr != br) & ~main
        if not (main.any() or anti.any()):
            return regions
        # Views alias `regions`; fixing one pinch can move a neighbour's,
        # which the next pass picks up.
        tr[main] = tl[main]
        tl[anti] = tr[anti]
    raise RuntimeError(f"pinches remain after {max_passes} passes")


def _corner_sum(padded: np.ndarray) -> np.ndarray:
    return padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]


def corner_average(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Averages per-pixel values onto pixel corners using only selected
    pixels, so the subject's edge is not pulled toward the background.
    Accepts (h, w) or (h, w, c) values; corners touching no selected pixel get 0."""
    weights = cells.astype(np.float32)
    if values.ndim == 3:
        weights = weights[..., None]
    padding = ((1, 1), (1, 1)) + ((0, 0),) * (values.ndim - 2)
    total = _corner_sum(np.pad(values * weights, padding))
    count = _corner_sum(np.pad(weights, padding))
    return total / np.maximum(count, 1)


def heightfield_mesh(corner_heights: np.ndarray, cells: np.ndarray, mm_per_pixel: float) -> Mesh:
    """Builds a closed solid over the selected pixels. Image row 0 maps to the
    far edge (largest y) so the model reads upright when viewed from above."""
    rows, cols = cells.shape
    stride = cols + 1
    top_count = (rows + 1) * stride
    ii, jj = np.nonzero(cells)
    a = ii * stride + jj  # top-left corner of each pixel
    b = a + 1
    c = a + stride
    d = c + 1

    faces = [
        np.stack([c, d, b], 1),
        np.stack([c, b, a], 1),
        np.stack([c, b, d], 1) + top_count,
        np.stack([c, a, b], 1) + top_count,
    ]
    padded = np.pad(cells, 1)
    # Each side of a pixel as a directed edge of its counter-clockwise top
    # face, with the neighbour across it. Open sides get a wall.
    sides = (
        (padded[ii + 2, jj + 1], c, d),
        (padded[ii + 1, jj + 2], d, b),
        (padded[ii, jj + 1], b, a),
        (padded[ii + 1, jj], a, c),
    )
    for neighbour, start, end in sides:
        start, end = start[~neighbour], end[~neighbour]
        faces.append(np.stack([start, start + top_count, end + top_count], 1))
        faces.append(np.stack([start, end + top_count, end], 1))

    used, inverse = np.unique(np.concatenate(faces), return_inverse=True)
    corners = used % top_count
    row, col = np.divmod(corners, stride)
    z = np.where(used < top_count, corner_heights.ravel()[corners], 0.0)
    vertices = np.column_stack([col * mm_per_pixel, (rows - row) * mm_per_pixel, z])
    return Mesh(vertices.astype(np.float32), inverse.reshape(-1, 3).astype(np.int32), corners)
