import math

from src.geometry.measurements import (
    compute_compactness,
    compute_edge_length,
    compute_face_area,
    compute_face_perimeter,
    get_measurement_fields,
    populate_face_measurements,
)


def test_get_measurement_fields():
    assert get_measurement_fields() == [
        "area",
        "perimeter",
        "compactness",
    ]


def test_compute_face_area_none_returns_zero():
    assert compute_face_area(None) == 0.0


def test_compute_edge_length_none_returns_zero():
    assert compute_edge_length(None) == 0.0


def test_compute_face_perimeter_none_returns_zero():
    assert compute_face_perimeter(None) == 0.0


def test_compute_face_perimeter_without_edges_returns_zero():
    class DummyFace:
        pass

    face = DummyFace()

    assert compute_face_perimeter(face) == 0.0


def test_compute_compactness_valid_circle_like_value():
    area = math.pi
    perimeter = 2.0 * math.pi

    compactness = compute_compactness(area, perimeter)

    assert math.isclose(compactness, 1.0, rel_tol=1e-9)


def test_compute_compactness_invalid_inputs_return_zero():
    assert compute_compactness(0.0, 10.0) == 0.0
    assert compute_compactness(10.0, 0.0) == 0.0
    assert compute_compactness(-1.0, 10.0) == 0.0
    assert compute_compactness(10.0, -1.0) == 0.0


def test_populate_face_measurements_rejects_none_model():
    try:
        populate_face_measurements(None)
    except ValueError as exc:
        assert "CADModel cannot be None" in str(exc)
    else:
        raise AssertionError("Expected ValueError for None CADModel.")


def test_populate_face_measurements_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    model = DummyModelWithoutFaces()

    try:
        populate_face_measurements(model)
    except AttributeError as exc:
        assert "faces" in str(exc)
    else:
        raise AssertionError("Expected AttributeError when CADModel has no faces.")


def test_populate_face_measurements_sets_fields_for_dummy_faces():
    class DummyFace:
        def __init__(self):
            self.shape = None
            self.edges = []

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace(), DummyFace()]

    model = DummyModel()

    measured_model = populate_face_measurements(model)

    assert measured_model is model

    for face in measured_model.faces:
        assert face.area == 0.0
        assert face.perimeter == 0.0
        assert face.compactness == 0.0