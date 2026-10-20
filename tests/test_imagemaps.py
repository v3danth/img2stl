import numpy as np

from imagemaps import (
    border_background,
    edge_mask,
    palette_labels,
    relief_heights,
    subject_mask,
    without_small_islands,
)


def test_relief_maps_dark_to_top_and_light_to_base():
    gray = np.array([[0, 255]], np.uint8)
    np.testing.assert_allclose(relief_heights(gray, 2.0, 4.0), [[6.0, 2.0]])


def test_relief_of_flat_image_is_base():
    np.testing.assert_allclose(relief_heights(np.full((2, 2), 9, np.uint8), 2.0, 4.0), 2.0)


def test_alpha_defines_subject():
    alpha = np.array([[0, 255], [127, 128]], np.uint8)
    bgr = np.zeros((2, 2, 3), np.uint8)
    np.testing.assert_array_equal(subject_mask(bgr, alpha), [[False, True], [False, True]])


def test_enclosed_background_colour_stays_in_subject():
    bgr = np.full((60, 60, 3), 255, np.uint8)
    bgr[10:50, 10:50] = 0
    bgr[20:40, 20:40] = 255  # white inside a black ring, like the Knight's mask
    background = border_background(bgr)
    assert background[0, 0]
    assert not background[30, 30]
    assert not background[12, 12]


def test_opaque_alpha_falls_back_to_border_colour():
    bgr = np.full((60, 60, 3), 255, np.uint8)
    bgr[20:40, 20:40] = 0
    mask = subject_mask(bgr, np.full((60, 60), 255, np.uint8))
    # Grain removal rounds the corners by a pixel or two.
    assert mask[22:38, 22:38].all() and abs(int(mask.sum()) - 400) <= 40


def test_grainy_background_is_removed():
    rng = np.random.default_rng(3)
    bgr = np.clip(rng.normal((76, 76, 242), 29, (120, 80, 3)), 0, 255).astype(np.uint8)
    bgr[30:90, 20:60] = 20  # dark subject
    bgr[45:75, 30:50] = 230  # light area inside it
    mask = subject_mask(bgr, None)
    assert mask[30:90, 20:60].mean() > 0.95
    # Allow the median filter's few-pixel fringe at the subject's edge.
    assert not (mask & ~np.pad(np.ones((66, 46), bool), ((27, 27), (17, 17)))).any()


def test_small_islands_are_dropped():
    mask = np.zeros((20, 20), bool)
    mask[2:12, 2:12] = True
    mask[16, 16] = True
    kept = without_small_islands(mask, 5)
    assert kept[2:12, 2:12].all() and not kept[16, 16]


def test_palette_splits_two_colours_darkest_first():
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:, 10:] = (250, 30, 30)
    labels, palette = palette_labels(rgb, np.ones((20, 20), bool), 4)
    assert len(palette) == 2
    assert palette[0].sum() < palette[1].sum()
    assert (labels[:, :8] == 0).all() and (labels[:, 12:] == 1).all()


def test_edges_trace_colour_boundaries_and_silhouette():
    rgb = np.full((30, 30, 3), 255, np.uint8)
    rgb[:, 15:] = 0
    subject = np.zeros((30, 30), bool)
    subject[5:25, 5:25] = True
    edges = edge_mask(rgb, subject, 3)
    assert edges[15, 14:16].all()  # the white/black boundary
    assert edges[5, 10] and edges[15, 5]  # the silhouette
    assert not edges[15, 10]  # flat colour inside
    assert not (edges & ~subject).any()
