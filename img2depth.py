"""Turns an image into 3D-printable models.

Outputs, in <image name>_3d/ next to the image unless --out-dir is given:
  relief.stl               the whole image as a slab, darker pixels raised
  cutout.stl               only the subject, same relief, no slab
  color_original.ply/.obj  the cut-out with the image's own colours (OBJ for Bambu Studio)
  color_palette.3mf        the cut-out split into a few flat colours, one part each
  color_palette_parts/     the same parts as separate STLs
  flat.stl                 the cut-out at one height, no relief
  flat_color_palette.3mf   the flat cut-out split into the same colours
  edges.stl                only the line art, at one height
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import exporters
import imagemaps
import meshing
from preview import show_preview

log = logging.getLogger("img2depth")

# Region value for pixels outside the subject; palette indices are below it.
BACKGROUND = 255


@dataclass(frozen=True)
class Settings:
    target_width: float = 100.0  # mm
    base_height: float = 2.0  # mm under the lightest pixel
    relief_height: float = 4.0  # mm added for the darkest pixel
    max_dimension: int = 1000  # pixels; larger images are scaled down
    palette_size: int = 4
    flat_height: float = 3.0  # mm, for the flat and edge models
    line_width: float = 0.8  # mm, for the edge model; about two nozzle widths


def save(writer: Callable[..., None], mesh: meshing.Mesh, path: Path, *extra) -> Path:
    start = time.perf_counter()
    writer(mesh, path, *extra)
    log.info("wrote %s triangles=%d seconds=%.1f", path.name, len(mesh.faces), time.perf_counter() - start)
    return path


def write_models(
    maps: imagemaps.ImageMaps,
    regions: np.ndarray,
    palette: np.ndarray,
    edges: np.ndarray,
    out_dir: Path,
    settings: Settings,
) -> list[Path]:
    rows, cols = regions.shape
    mm_per_pixel = settings.target_width / cols
    subject = regions != BACKGROUND
    whole = np.ones_like(subject)
    parts_dir = out_dir / "color_palette_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    flat_heights = np.full((rows + 1, cols + 1), settings.flat_height, np.float32)

    relief = meshing.heightfield_mesh(meshing.corner_average(maps.heights, whole), whole, mm_per_pixel)
    outputs = [save(exporters.write_stl, relief, out_dir / "relief.stl")]

    # Every subject-shaped model shares these corner heights, so palette
    # parts meet the cut-out's surface and each other without seams.
    subject_heights = meshing.corner_average(maps.heights, subject)
    cutout = meshing.heightfield_mesh(subject_heights, subject, mm_per_pixel)
    outputs.append(save(exporters.write_stl, cutout, out_dir / "cutout.stl"))

    corner_rgb = meshing.corner_average(maps.rgb.astype(np.float32), subject).reshape(-1, 3)
    vertex_rgb = np.rint(corner_rgb[cutout.corners]).astype(np.uint8)
    outputs.append(save(exporters.write_ply, cutout, out_dir / "color_original.ply", vertex_rgb))
    outputs.append(save(exporters.write_obj, cutout, out_dir / "color_original.obj", vertex_rgb))

    outputs += write_palette(regions, palette, subject_heights, mm_per_pixel, out_dir / "color_palette.3mf", parts_dir)

    flat = meshing.heightfield_mesh(flat_heights, subject, mm_per_pixel)
    outputs.append(save(exporters.write_stl, flat, out_dir / "flat.stl"))
    outputs += write_palette(regions, palette, flat_heights, mm_per_pixel, out_dir / "flat_color_palette.3mf")

    outputs.append(
        save(exporters.write_stl, meshing.heightfield_mesh(flat_heights, edges, mm_per_pixel), out_dir / "edges.stl")
    )

    log.info(
        "model size width_mm=%.1f depth_mm=%.1f max_height_mm=%.1f",
        cols * mm_per_pixel,
        rows * mm_per_pixel,
        float(maps.heights.max()),
    )
    return outputs


def edge_cells(maps: imagemaps.ImageMaps, subject: np.ndarray, line_px: float) -> np.ndarray:
    return meshing.without_pinches(imagemaps.edge_mask(maps.rgb, subject, line_px).astype(np.uint8)) == 1


def write_palette(
    regions: np.ndarray,
    palette: np.ndarray,
    corner_heights: np.ndarray,
    mm_per_pixel: float,
    path: Path,
    parts_dir: Path | None = None,
) -> list[Path]:
    """Writes one coloured part per palette entry to a 3MF, and to separate
    STLs as well when parts_dir is given."""
    outputs, parts = [], []
    for index, rgb in enumerate(palette):
        cells = regions == index
        if not cells.any():
            continue
        name = f"color_{index + 1}_{exporters.rgb_to_hex(rgb)[1:]}"
        part = meshing.heightfield_mesh(corner_heights, cells, mm_per_pixel)
        if parts_dir is not None:
            outputs.append(save(exporters.write_stl, part, parts_dir / f"{name}.stl"))
        parts.append(exporters.ColoredPart(name, part, rgb))
    start = time.perf_counter()
    exporters.write_3mf(parts, path)
    log.info("wrote %s parts=%d seconds=%.1f", path.name, len(parts), time.perf_counter() - start)
    return [*outputs, path]


def preview_panels(
    maps: imagemaps.ImageMaps, regions: np.ndarray, palette: np.ndarray, edges: np.ndarray
) -> list[tuple[str, np.ndarray]]:
    span = float(np.ptp(maps.heights)) or 1.0
    depth = ((maps.heights - maps.heights.min()) / span * 255).astype(np.uint8)
    depth = cv2.cvtColor(cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    white = np.full_like(maps.rgb, 255)
    subject = (regions != BACKGROUND)[..., None]
    flat = palette[np.minimum(regions, len(palette) - 1)]
    return [
        ("Original", maps.rgb),
        ("Depth", depth),
        ("Cut-out", np.where(subject, maps.rgb, white)),
        (f"Palette ({len(palette)} colours)", np.where(subject, flat, white)),
        ("Edges", np.where(edges[..., None], 0, white).astype(np.uint8)),
    ]


def positive(kind: type) -> Callable[[str], float]:
    def parse(text: str):
        value = kind(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be positive, got {text}")
        return value

    return parse


def parse_args(argv: list[str]) -> argparse.Namespace:
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Convert an image into 3D-printable models.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-dir", type=Path, help="default: <image dir>/<image name>_3d")
    parser.add_argument("--width", type=positive(float), default=defaults.target_width, help="model width in mm")
    parser.add_argument("--base", type=positive(float), default=defaults.base_height, help="base height in mm")
    parser.add_argument("--relief", type=positive(float), default=defaults.relief_height, help="relief depth in mm")
    parser.add_argument("--max-dimension", type=positive(int), default=defaults.max_dimension)
    parser.add_argument("--colors", type=positive(int), default=defaults.palette_size, help="palette size")
    parser.add_argument(
        "--flat-height", type=positive(float), default=defaults.flat_height, help="flat and edge model height in mm"
    )
    parser.add_argument("--line-width", type=positive(float), default=defaults.line_width, help="edge line width in mm")
    parser.add_argument("--no-preview", action="store_true", help="skip the preview window")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    if args.colors >= BACKGROUND:
        log.error("--colors must be below %d, got %d", BACKGROUND, args.colors)
        return 2
    settings = Settings(
        args.width, args.base, args.relief, args.max_dimension, args.colors, args.flat_height, args.line_width
    )
    out_dir = args.out_dir or args.image.with_name(f"{args.image.stem}_3d")
    try:
        maps = imagemaps.image_maps(args.image, settings.max_dimension, settings.base_height, settings.relief_height)
    except ValueError as err:
        log.error("%s", err)
        return 1
    labels, palette = imagemaps.palette_labels(maps.rgb, maps.subject, settings.palette_size)
    regions = meshing.without_pinches(np.where(maps.subject, labels, BACKGROUND).astype(np.uint8))
    line_px = settings.line_width * regions.shape[1] / settings.target_width
    edges = edge_cells(maps, regions != BACKGROUND, line_px)
    write_models(maps, regions, palette, edges, out_dir, settings)
    log.info("done out_dir=%s", out_dir)
    if not args.no_preview:
        show_preview(preview_panels(maps, regions, palette, edges))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
