from src.geometry.surface_classifier import (
    classify_model_surfaces,
    classify_surface_type,
    get_supported_surface_types,
)


def test_supported_surface_types_contains_expected_labels():
    surface_types = get_supported_surface_types()

    expected = {
        "PLANE",
        "CYLINDER",
        "CONE",
        "SPHERE",
        "TORUS",
        "BSPLINE",
        "BEZIER",
        "OFFSET",
        "REVOLUTION",
        "EXTRUSION",
        "OTHER",
        "UNKNOWN",
    }

    assert set(surface_types) == expected


def test_classify_surface_type_none_returns_unknown():
    assert classify_surface_type(None) == "UNKNOWN"


def test_classify_model_surfaces_rejects_none_model():
    try:
        classify_model_surfaces(None)
    except ValueError as exc:
        assert "CADModel cannot be None" in str(exc)
    else:
        raise AssertionError("Expected ValueError for None CADModel.")


def test_classify_model_surfaces_populates_face_surface_type():
    class DummyFace:
        def __init__(self):
            self.shape = None
            self.surface_type = None

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace(), DummyFace()]

    model = DummyModel()

    classified_model = classify_model_surfaces(model)

    assert classified_model is model
    assert len(classified_model.faces) == 2
    assert classified_model.faces[0].surface_type == "UNKNOWN"
    assert classified_model.faces[1].surface_type == "UNKNOWN"


def test_classify_model_surfaces_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    model = DummyModelWithoutFaces()

    try:
        classify_model_surfaces(model)
    except AttributeError as exc:
        assert "faces" in str(exc)
    else:
        raise AssertionError("Expected AttributeError when CADModel has no faces.")