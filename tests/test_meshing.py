import numpy as np
import pytest

from meshing import corner_average, heightfield_mesh, without_pinches


def directed_edges(faces: np.ndarray) -> np.ndarray:
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def assert_closed(mesh) -> None:
    """Closed and consistently oriented: every directed edge appears once and
    its reverse appears once."""
    edges = {tuple(edge) for edge in directed_edges(mesh.faces).tolist()}
    assert len(edges) == 3 * len(mesh.faces)
    assert all((end, start) in edges for start, end in edges)


def signed_volume(mesh) -> float:
    v0, v1, v2 = (mesh.vertices[mesh.faces[:, k]].astype(np.float64) for k in range(3))
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6)


# A ring with a hole and a separate island, to cover inner walls and islands.
SHAPE = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 0, 0],
    ],
    dtype=bool,
)


def test_mesh_is_closed_and_outward():
    heights = np.random.default_rng(0).uniform(1, 3, SHAPE.shape).astype(np.float32)
    mesh = heightfield_mesh(corner_average(heights, SHAPE), SHAPE, 0.5)
    assert_closed(mesh)
    assert signed_volume(mesh) > 0


def test_flat_volume_is_cells_times_height():
    heights = np.full(SHAPE.shape, 2.0, np.float32)
    mesh = heightfield_mesh(corner_average(heights, SHAPE), SHAPE, 0.5)
    assert signed_volume(mesh) == pytest.approx(SHAPE.sum() * 0.25 * 2.0)


def test_corner_average_ignores_unselected_pixels():
    heights = np.array([[2.0, 100.0]], np.float32)
    cells = np.array([[True, False]])
    corners = corner_average(heights, cells)
    np.testing.assert_allclose(corners[:, :2], 2.0)
    np.testing.assert_allclose(corners[:, 2], 0.0)


def test_image_top_row_is_at_far_edge():
    cells = np.array([[True], [False]])
    mesh = heightfield_mesh(np.ones((3, 2), np.float32), cells, 1.0)
    assert mesh.vertices[:, 1].min() == pytest.approx(1.0)
    assert mesh.vertices[:, 1].max() == pytest.approx(2.0)


def test_adjacent_parts_share_their_seam():
    heights = np.random.default_rng(1).uniform(1, 3, (2, 4)).astype(np.float32)
    whole = np.ones((2, 4), bool)
    shared = corner_average(heights, whole)
    left, right = whole.copy(), whole.copy()
    left[:, 2:] = False
    right[:, :2] = False
    left_mesh = heightfield_mesh(shared, left, 1.0)
    right_mesh = heightfield_mesh(shared, right, 1.0)
    seam = lambda mesh: {tuple(v) for v in mesh.vertices.tolist() if v[0] == 2.0}
    assert seam(left_mesh) == seam(right_mesh)


def pinch_count(regions: np.ndarray) -> int:
    tl, tr, bl, br = regions[:-1, :-1], regions[:-1, 1:], regions[1:, :-1], regions[1:, 1:]
    return int((((tl == br) & (tl != tr) & (tl != bl)) | ((tr == bl) & (tr != tl) & (tr != br))).sum())


def test_without_pinches_closes_every_region():
    regions = np.random.default_rng(2).integers(0, 3, (40, 40)).astype(np.uint8)
    assert pinch_count(regions) > 0
    fixed = without_pinches(regions)
    assert pinch_count(fixed) == 0
    for value in np.unique(fixed):
        cells = fixed == value
        assert_closed(heightfield_mesh(corner_average(np.ones(cells.shape, np.float32), cells), cells, 1.0))


def test_without_pinches_keeps_clean_regions():
    regions = np.array([[0, 0, 1], [0, 1, 1]], np.uint8)
    np.testing.assert_array_equal(without_pinches(regions), regions)
