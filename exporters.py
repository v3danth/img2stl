"""Writers for the output formats. STL carries shape only; PLY and OBJ
carry per-vertex colour (Bambu Studio reads OBJ colours, not PLY); 3MF
carries named, coloured parts that slicers load as one multi-material object."""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from stl import mesh as stl_mesh

from meshing import Mesh

# Rows formatted per write, to bound memory when serialising large meshes as text.
TEXT_CHUNK_ROWS = 100_000

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""
_RELATIONSHIPS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""
_MODEL_OPEN = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
<resources>
"""


@dataclass(frozen=True)
class ColoredPart:
    name: str
    mesh: Mesh
    rgb: np.ndarray  # (3,) uint8


def rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(v) for v in rgb))


def write_stl(mesh: Mesh, path: Path) -> None:
    solid = stl_mesh.Mesh(np.zeros(len(mesh.faces), dtype=stl_mesh.Mesh.dtype))
    solid.vectors[:] = mesh.vertices[mesh.faces]
    solid.update_normals()
    solid.save(str(path))


def write_ply(mesh: Mesh, path: Path, vertex_rgb: np.ndarray) -> None:
    vertices = np.empty(len(mesh.vertices), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    vertices["xyz"] = mesh.vertices
    vertices["rgb"] = vertex_rgb
    faces = np.empty(len(mesh.faces), dtype=[("count", "u1"), ("ids", "<i4", 3)])
    faces["count"] = 3
    faces["ids"] = mesh.faces
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as out:
        out.write(header.encode("ascii"))
        out.write(vertices.tobytes())
        out.write(faces.tobytes())


def write_obj(mesh: Mesh, path: Path, vertex_rgb: np.ndarray) -> None:
    """Uses the common "v x y z r g b" vertex-colour extension, colours in 0..1."""
    rows = np.column_stack([mesh.vertices, vertex_rgb / 255.0])
    with open(path, "w", encoding="ascii") as out:
        _write_rows(out, "v %.4f %.4f %.4f %.4f %.4f %.4f\n", rows)
        _write_rows(out, "f %d %d %d\n", mesh.faces + 1)


def _write_rows(out: io.TextIOBase, template: str, rows: np.ndarray) -> None:
    for start in range(0, len(rows), TEXT_CHUNK_ROWS):
        chunk = rows[start : start + TEXT_CHUNK_ROWS].tolist()
        out.write("".join(template % tuple(row) for row in chunk))


def _write_model(out: io.TextIOBase, parts: list[ColoredPart]) -> None:
    out.write(_MODEL_OPEN)
    out.write('<basematerials id="1">\n')
    out.writelines(f'<base name={quoteattr(part.name)} displaycolor="{rgb_to_hex(part.rgb)}"/>\n' for part in parts)
    out.write("</basematerials>\n")
    for index, part in enumerate(parts):
        out.write(f'<object id="{index + 2}" type="model" name={quoteattr(part.name)} pid="1" pindex="{index}">')
        out.write("<mesh><vertices>")
        _write_rows(out, '<vertex x="%.4f" y="%.4f" z="%.4f"/>', part.mesh.vertices)
        out.write("</vertices><triangles>")
        _write_rows(out, '<triangle v1="%d" v2="%d" v3="%d"/>', part.mesh.faces)
        out.write("</triangles></mesh></object>\n")
    # One assembly object, so slicers treat the colours as parts of a single model.
    assembly = len(parts) + 2
    out.write(f'<object id="{assembly}" type="model" name="assembly"><components>')
    out.write("".join(f'<component objectid="{index + 2}"/>' for index in range(len(parts))))
    out.write(f'</components></object>\n</resources>\n<build><item objectid="{assembly}"/></build>\n</model>\n')


def _bambu_model_settings(parts: list[ColoredPart]) -> str:
    """Bambu Studio ignores 3MF colours and reads each part's filament slot
    from this file instead. Slot n holds palette colour n. Only this file is
    written: marking the package as Bambu-made would make Bambu also expect
    its full project settings (a partial set crashed Bambu Studio 2.8)."""
    assembly = len(parts) + 2
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>", f'  <object id="{assembly}">']
    lines.append('    <metadata key="name" value="assembly"/>')
    lines.append('    <metadata key="extruder" value="1"/>')
    for index, part in enumerate(parts):
        lines.append(f'    <part id="{index + 2}" subtype="normal_part">')
        lines.append(f'      <metadata key="name" value={quoteattr(part.name)}/>')
        lines.append(f'      <metadata key="extruder" value="{index + 1}"/>')
        lines.append("    </part>")
    lines += ["  </object>", "</config>", ""]
    return "\n".join(lines)


def write_3mf(parts: list[ColoredPart], path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", _CONTENT_TYPES)
        package.writestr("_rels/.rels", _RELATIONSHIPS)
        package.writestr("Metadata/model_settings.config", _bambu_model_settings(parts))
        with (
            package.open("3D/3dmodel.model", "w", force_zip64=True) as raw,
            io.TextIOWrapper(raw, encoding="utf-8") as model,
        ):
            _write_model(model, parts)
