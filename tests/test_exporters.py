import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from stl import mesh as stl_mesh

from exporters import ColoredPart, write_3mf, write_obj, write_ply, write_stl
from meshing import heightfield_mesh

CELLS = np.array([[True, True], [False, True]])
MESH = heightfield_mesh(np.ones((3, 3), np.float32), CELLS, 1.0)
NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def test_stl_round_trip(tmp_path):
    write_stl(MESH, tmp_path / "a.stl")
    loaded = stl_mesh.Mesh.from_file(str(tmp_path / "a.stl"))
    np.testing.assert_allclose(loaded.vectors, MESH.vertices[MESH.faces])


def test_ply_header_and_size(tmp_path):
    rgb = np.zeros((len(MESH.vertices), 3), np.uint8)
    write_ply(MESH, tmp_path / "a.ply", rgb)
    raw = (tmp_path / "a.ply").read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert f"element vertex {len(MESH.vertices)}".encode() in header
    assert f"element face {len(MESH.faces)}".encode() in header
    assert len(body) == len(MESH.vertices) * 15 + len(MESH.faces) * 13


def test_3mf_has_one_coloured_object_per_part(tmp_path):
    parts = [ColoredPart("dark", MESH, np.array([0, 0, 0])), ColoredPart("red", MESH, np.array([255, 0, 0]))]
    write_3mf(parts, tmp_path / "a.3mf")
    with zipfile.ZipFile(tmp_path / "a.3mf") as package:
        model = ET.fromstring(package.read("3D/3dmodel.model"))
    colours = [base.get("displaycolor") for base in model.iterfind(".//m:base", NS)]
    assert colours == ["#000000", "#FF0000"]
    meshes = model.findall(".//m:object/m:mesh", NS)
    assert len(meshes) == 2
    assert len(meshes[0].findall(".//m:triangle", NS)) == len(MESH.faces)
    assert len(model.findall(".//m:component", NS)) == 2


def test_obj_has_coloured_vertices_and_one_based_faces(tmp_path):
    rgb = np.full((len(MESH.vertices), 3), 255, np.uint8)
    write_obj(MESH, tmp_path / "a.obj", rgb)
    lines = (tmp_path / "a.obj").read_text().splitlines()
    vertices = [line.split() for line in lines if line.startswith("v ")]
    faces = np.array([line.split()[1:] for line in lines if line.startswith("f ")], int)
    assert len(vertices) == len(MESH.vertices) and all(v[4:] == ["1.0000"] * 3 for v in vertices)
    np.testing.assert_array_equal(faces, MESH.faces + 1)


def test_3mf_assigns_each_part_its_own_bambu_filament(tmp_path):
    parts = [ColoredPart("dark", MESH, np.array([0, 0, 0])), ColoredPart("red", MESH, np.array([255, 0, 0]))]
    write_3mf(parts, tmp_path / "a.3mf")
    with zipfile.ZipFile(tmp_path / "a.3mf") as package:
        config = ET.fromstring(package.read("Metadata/model_settings.config"))
        model = ET.fromstring(package.read("3D/3dmodel.model"))
    assembly = config.find("object")
    assert assembly.get("id") == model.findall(".//m:object", NS)[-1].get("id")
    slots = {part.get("id"): part.find("metadata[@key='extruder']").get("value") for part in assembly.iter("part")}
    assert slots == {"2": "1", "3": "2"}
