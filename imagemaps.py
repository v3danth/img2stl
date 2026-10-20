"""Per-pixel maps derived from the input image: colours, relief heights,
the subject silhouette and a reduced colour palette."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Pixels at least this opaque belong to the subject when the image has alpha.
OPAQUE_ALPHA = 128
# Without usable alpha, background is whatever stays within a Lab distance
# of the border colour. The tolerance follows how much the border itself
# varies (film grain, gradients), never below the floor.
MIN_BACKGROUND_DISTANCE = 12.0
BORDER_SPREAD_PERCENTILE = 95
BORDER_SPREAD_MARGIN = 1.5
# Median kernel that removes grain before colours are compared.
DENOISE_KERNEL = 7
# Islands smaller than this fraction of the image are noise, not subject.
MIN_SUBJECT_FRACTION = 0.001
# The relief is computed at this fraction of the working resolution and
# scaled back up, which removes detail too fine to print.
DETAIL_SCALE = 0.25
SMOOTHING_KERNEL = 5


@dataclass(frozen=True)
class ImageMaps:
    rgb: np.ndarray  # (h, w, 3) uint8, original colours
    heights: np.ndarray  # (h, w) float32, top surface height in mm
    subject: np.ndarray  # (h, w) bool, True inside the subject


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"unsupported pixel type {image.dtype} in {path}")
    return image


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns the BGR colour planes and the alpha plane, if there is one."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        return np.ascontiguousarray(image[..., :3]), np.ascontiguousarray(image[..., 3])
    return image, None


def fit_within(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return image
    scale = max_dimension / max(height, width)
    return cv2.resize(image, (int(width * scale), int(height * scale)))


def smoothed_gray(gray: np.ndarray) -> np.ndarray:
    """Drops fine detail and speckle so the relief prints cleanly."""
    height, width = gray.shape
    coarse = cv2.resize(gray, (int(width * DETAIL_SCALE), int(height * DETAIL_SCALE)))
    smooth = cv2.resize(coarse, (width, height))
    kernel = np.ones((SMOOTHING_KERNEL, SMOOTHING_KERNEL), np.uint8)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(smooth, (SMOOTHING_KERNEL, SMOOTHING_KERNEL), 0)


def relief_heights(gray: np.ndarray, base_height: float, relief_height: float) -> np.ndarray:
    """Maps the darkest pixel to base + relief and the lightest to base."""
    lightest, darkest = float(gray.max()), float(gray.min())
    span = lightest - darkest
    if span == 0:
        return np.full(gray.shape, base_height, np.float32)
    darkness = (lightest - gray.astype(np.float32)) / span
    return (base_height + relief_height * darkness).astype(np.float32)


def without_small_islands(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Drops 8-connected islands smaller than min_area pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # label 0 is the area outside every island
    return keep[labels] if count > 1 else mask.copy()


def _border_pixels(image: np.ndarray) -> np.ndarray:
    return np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])


def border_background(bgr: np.ndarray) -> np.ndarray:
    """Background is the border-coloured region connected to the image edge,
    so areas of the same colour enclosed by the subject are kept."""
    lab = cv2.cvtColor(cv2.medianBlur(bgr, DENOISE_KERNEL), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = _border_pixels(lab)
    colour = np.median(border, axis=0)
    spread = np.percentile(np.linalg.norm(border - colour, axis=1), BORDER_SPREAD_PERCENTILE)
    tolerance = max(MIN_BACKGROUND_DISTANCE, spread * BORDER_SPREAD_MARGIN)
    near = (np.linalg.norm(lab - colour, axis=2) <= tolerance).astype(np.uint8)
    _, labels = cv2.connectedComponents(near, connectivity=4)
    edge = np.unique(_border_pixels(labels))
    # Component 0 is everything not near the border colour.
    return np.isin(labels, edge[edge != 0])


def subject_mask(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is not None and (alpha < OPAQUE_ALPHA).any():
        mask = alpha >= OPAQUE_ALPHA
    else:
        mask = ~border_background(bgr)
    mask = without_small_islands(mask, MIN_SUBJECT_FRACTION * mask.size)
    # An image that is all background has no subject to cut out; keep it whole.
    return mask if mask.any() else np.ones_like(mask)


def image_maps(path: Path, max_dimension: int, base_height: float, relief_height: float) -> ImageMaps:
    bgr, alpha = split_alpha(read_image(path))
    bgr = fit_within(bgr, max_dimension)
    if alpha is not None:
        alpha = fit_within(alpha, max_dimension)
    gray = smoothed_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return ImageMaps(
        rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        heights=relief_heights(gray, base_height, relief_height),
        subject=subject_mask(bgr, alpha),
    )


def edge_mask(rgb: np.ndarray, subject: np.ndarray, line_px: float) -> np.ndarray:
    """Line art of the subject: colour edges plus the silhouette, widened to
    line_px so every line is thick enough to print. Lines shorter than a few
    pen widths are grain or texture, and would print as loose specks."""
    gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), SMOOTHING_KERNEL)
    edges = cv2.Canny(gray, 50, 150) > 0
    # The silhouette needs adding explicitly: an outline drawn in the same
    # colour as the background has no colour edge to detect.
    ring = np.ones((3, 3), np.uint8)
    silhouette = cv2.morphologyEx(subject.astype(np.uint8), cv2.MORPH_GRADIENT, ring) > 0
    size = max(1, round(line_px))
    pen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    lines = (cv2.dilate((edges | silhouette).astype(np.uint8), pen) > 0) & subject
    return without_small_islands(lines, 4 * size * size)


def palette_labels(rgb: np.ndarray, subject: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduces the subject to at most `size` flat colours so each can print
    with its own filament. Returns per-pixel palette indices and the palette
    as RGB, ordered darkest first so file names are stable between runs."""
    lab = cv2.cvtColor(cv2.medianBlur(rgb, SMOOTHING_KERNEL), cv2.COLOR_RGB2LAB)
    samples = lab[subject].astype(np.float32)
    count = min(size, len(np.unique(samples, axis=0)))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(samples, count, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centres = centres[np.argsort(centres[:, 0])]
    distances = ((lab.astype(np.float32)[:, :, None, :] - centres) ** 2).sum(axis=3)
    labels = cv2.medianBlur(distances.argmin(axis=2).astype(np.uint8), SMOOTHING_KERNEL)
    palette = cv2.cvtColor(np.clip(centres, 0, 255).astype(np.uint8)[None], cv2.COLOR_LAB2RGB)[0]
    return labels, palette
