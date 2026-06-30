import pytest

from src.knowledge.feature_extractor import (
    attach_features_to_model,
    extract_face_features,
    extract_model_features,
    get_feature_names,
    id_to_surface_type,
    safe_float,
    safe_int,
    surface_type_to_id,
)


def test_surface_type_to_id_and_id_to_surface_type():
    assert surface_type_to_id("PLANE") == 0
    assert surface_type_to_id("cylinder") == 1
    assert surface_type_to_id("invalid") == 11
    assert surface_type_to_id(None) == 11

    assert id_to_surface_type(0) == "PLANE"
    assert id_to_surface_type(1) == "CYLINDER"
    assert id_to_surface_type(99) == "UNKNOWN"


def test_safe_float():
    assert safe_float("10.5") == 10.5
    assert safe_float(None) == 0.0
    assert safe_float("bad") == 0.0
    assert safe_float(float("nan")) == 0.0
    assert safe_float("bad", default=2.5) == 2.5


def test_safe_int():
    assert safe_int("10") == 10
    assert safe_int(3.8) == 3
    assert safe_int(None) == 0
    assert safe_int("bad") == 0
    assert safe_int("bad", default=5) == 5


def test_get_feature_names():
    assert get_feature_names() == [
        "face_id",
        "surface_type",
        "surface_type_id",
        "area",
        "perimeter",
        "compactness",
        "edge_count",
        "neighbor_count",
        "is_analytical",
        "is_freeform",
        "is_curved",
        "is_high_complexity_surface",
    ]


def test_extract_face_features_complete_face():
    class DummyFace:
        def __init__(self):
            self.face_id = 7
            self.surface_type = "CYLINDER"
            self.area = 100.0
            self.perimeter = 50.0
            self.compactness = 0.5
            self.edge_count = 4
            self.neighbor_count = 3
            self.edges = [object(), object()]
            self.neighbors = [1, 2]

    features = extract_face_features(DummyFace())

    assert features["face_id"] == 7
    assert features["surface_type"] == "CYLINDER"
    assert features["surface_type_id"] == 1
    assert features["area"] == 100.0
    assert features["perimeter"] == 50.0
    assert features["compactness"] == 0.5
    assert features["edge_count"] == 4
    assert features["neighbor_count"] == 3
    assert features["is_analytical"] is True
    assert features["is_freeform"] is False
    assert features["is_curved"] is True
    assert features["is_high_complexity_surface"] is False


def test_extract_face_features_uses_fallback_face_id_and_counts():
    class DummyFace:
        def __init__(self):
            self.surface_type = "BSPLINE"
            self.area = "20.0"
            self.perimeter = "15.0"
            self.compactness = "0.4"
            self.edges = [object(), object(), object()]
            self.neighbors = [1, 2]

    features = extract_face_features(DummyFace(), face_id=3)

    assert features["face_id"] == 3
    assert features["surface_type"] == "BSPLINE"
    assert features["surface_type_id"] == 5
    assert features["area"] == 20.0
    assert features["perimeter"] == 15.0
    assert features["compactness"] == 0.4
    assert features["edge_count"] == 3
    assert features["neighbor_count"] == 2
    assert features["is_analytical"] is False
    assert features["is_freeform"] is True
    assert features["is_curved"] is True
    assert features["is_high_complexity_surface"] is True


def test_extract_face_features_none_raises():
    with pytest.raises(ValueError, match="Face cannot be None"):
        extract_face_features(None)


def test_extract_model_features():
    class DummyFace:
        def __init__(self, surface_type):
            self.surface_type = surface_type
            self.area = 1.0
            self.perimeter = 4.0
            self.compactness = 0.7
            self.edges = []
            self.neighbors = []

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace("PLANE"), DummyFace("TORUS")]

    features = extract_model_features(DummyModel())

    assert len(features) == 2
    assert features[0]["face_id"] == 0
    assert features[0]["surface_type"] == "PLANE"
    assert features[1]["face_id"] == 1
    assert features[1]["surface_type"] == "TORUS"
    assert features[1]["is_high_complexity_surface"] is True


def test_extract_model_features_rejects_none_model():
    with pytest.raises(ValueError, match="CADModel cannot be None"):
        extract_model_features(None)


def test_extract_model_features_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    with pytest.raises(AttributeError, match="faces"):
        extract_model_features(DummyModelWithoutFaces())


def test_attach_features_to_model():
    class DummyFace:
        def __init__(self, surface_type):
            self.surface_type = surface_type
            self.area = 2.0
            self.perimeter = 6.0
            self.compactness = 0.6
            self.edges = []
            self.neighbors = []

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace("PLANE"), DummyFace("CYLINDER")]

    model = DummyModel()
    returned = attach_features_to_model(model)

    assert returned is model
    assert model.faces[0].features["surface_type"] == "PLANE"
    assert model.faces[1].features["surface_type"] == "CYLINDER"
    assert model.faces[1].features["is_curved"] is True


def test_attach_features_to_model_rejects_none_model():
    with pytest.raises(ValueError, match="CADModel cannot be None"):
        attach_features_to_model(None)


def test_attach_features_to_model_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    with pytest.raises(AttributeError, match="faces"):
        attach_features_to_model(DummyModelWithoutFaces())