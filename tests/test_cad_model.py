import pytest

from src.geometry.cad_model import CADModel, EdgeData, FaceData, VertexData


def test_cad_model_counts():
    model = CADModel(
        model_id="CM0001",
        file_name="CM0001.step",
        faces=[FaceData(face_id=0), FaceData(face_id=1)],
        edges=[EdgeData(edge_id=0)],
        vertices=[VertexData(vertex_id=0, coordinates=(0.0, 0.0, 0.0))],
    )

    assert model.face_count == 2
    assert model.edge_count == 1
    assert model.vertex_count == 1


def test_get_face_by_id():
    model = CADModel(
        model_id="CM0001",
        file_name="CM0001.step",
        faces=[FaceData(face_id=7, surface_type="PLANE")],
    )

    face = model.get_face(7)

    assert face.face_id == 7
    assert face.surface_type == "PLANE"


def test_get_face_invalid_id_raises_error():
    model = CADModel(model_id="CM0001", file_name="CM0001.step")

    with pytest.raises(KeyError):
        model.get_face(99)