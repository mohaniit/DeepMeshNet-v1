import pytest

from src.knowledge.feature_vector import (
    FaceFeatureVector,
    attach_feature_vectors_to_model,
    bool_to_float,
    build_face_feature_vector,
    build_model_feature_vectors,
    build_xy,
    get_feature_vector_names,
    safe_numeric,
)


def test_feature_vector_names():
    assert get_feature_vector_names() == [
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
        "complexity_score",
    ]


def test_face_feature_vector_dataclass():
    vector = FaceFeatureVector(
        face_id=1,
        x=[1.0, 2.0],
        y=3,
        density_label="VERY_HIGH",
    )

    assert vector.face_id == 1
    assert vector.x == [1.0, 2.0]
    assert vector.y == 3
    assert vector.density_label == "VERY_HIGH"


def test_safe_numeric():
    assert safe_numeric(1) == 1.0
    assert safe_numeric("2.5") == 2.5
    assert safe_numeric(None) == 0.0
    assert safe_numeric("bad") == 0.0
    assert safe_numeric(float("nan")) == 0.0
    assert safe_numeric(float("inf")) == 0.0
    assert safe_numeric(float("-inf")) == 0.0
    assert safe_numeric("bad", default=7.0) == 7.0


def test_bool_to_float():
    assert bool_to_float(True) == 1.0
    assert bool_to_float(False) == 0.0
    assert bool_to_float(1) == 1.0
    assert bool_to_float(0) == 0.0
    assert bool_to_float("") == 0.0
    assert bool_to_float("yes") == 1.0


def test_build_face_feature_vector_complete_face():
    class DummyFace:
        def __init__(self):
            self.features = {
                "face_id": 5,
                "surface_type_id": 1,
                "area": 100.0,
                "perimeter": 50.0,
                "compactness": 0.7,
                "edge_count": 4,
                "neighbor_count": 3,
                "is_analytical": True,
                "is_freeform": False,
                "is_curved": True,
                "is_high_complexity_surface": False,
            }
            self.complexity_score = 0.42
            self.density_id = 1
            self.density_label = "MEDIUM"

    vector = build_face_feature_vector(DummyFace())

    assert vector.face_id == 5
    assert vector.y == 1
    assert vector.density_label == "MEDIUM"
    assert vector.x == [
        1.0,
        100.0,
        50.0,
        0.7,
        4.0,
        3.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.42,
    ]


def test_build_face_feature_vector_missing_optional_label():
    class DummyFace:
        def __init__(self):
            self.features = {
                "face_id": 0,
                "surface_type_id": 0,
                "area": 1.0,
                "perimeter": 4.0,
                "compactness": 0.9,
                "edge_count": 4,
                "neighbor_count": 2,
                "is_analytical": True,
                "is_freeform": False,
                "is_curved": False,
                "is_high_complexity_surface": False,
            }
            self.complexity_score = 0.1

    vector = build_face_feature_vector(DummyFace())

    assert vector.face_id == 0
    assert vector.y is None
    assert vector.density_label is None
    assert len(vector.x) == 11


def test_build_face_feature_vector_rejects_none_face():
    with pytest.raises(ValueError, match="Face cannot be None"):
        build_face_feature_vector(None)


def test_build_face_feature_vector_requires_features():
    class DummyFace:
        def __init__(self):
            self.complexity_score = 0.1

    with pytest.raises(AttributeError, match="features"):
        build_face_feature_vector(DummyFace())


def test_build_face_feature_vector_requires_complexity_score():
    class DummyFace:
        def __init__(self):
            self.features = {"face_id": 0}

    with pytest.raises(AttributeError, match="complexity_score"):
        build_face_feature_vector(DummyFace())


def test_build_model_feature_vectors():
    class DummyFace:
        def __init__(self, face_id):
            self.features = {
                "face_id": face_id,
                "surface_type_id": 0,
                "area": 1.0,
                "perimeter": 4.0,
                "compactness": 0.8,
                "edge_count": 4,
                "neighbor_count": 2,
                "is_analytical": True,
                "is_freeform": False,
                "is_curved": False,
                "is_high_complexity_surface": False,
            }
            self.complexity_score = 0.2
            self.density_id = 0
            self.density_label = "LOW"

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace(0), DummyFace(1)]

    vectors = build_model_feature_vectors(DummyModel())

    assert len(vectors) == 2
    assert vectors[0].face_id == 0
    assert vectors[1].face_id == 1


def test_build_model_feature_vectors_rejects_none_model():
    with pytest.raises(ValueError, match="CADModel cannot be None"):
        build_model_feature_vectors(None)


def test_build_model_feature_vectors_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    with pytest.raises(AttributeError, match="faces"):
        build_model_feature_vectors(DummyModelWithoutFaces())


def test_build_xy():
    class DummyFace:
        def __init__(self, face_id, density_id):
            self.features = {
                "face_id": face_id,
                "surface_type_id": 0,
                "area": 1.0,
                "perimeter": 4.0,
                "compactness": 0.8,
                "edge_count": 4,
                "neighbor_count": 2,
                "is_analytical": True,
                "is_freeform": False,
                "is_curved": False,
                "is_high_complexity_surface": False,
            }
            self.complexity_score = 0.2
            self.density_id = density_id
            self.density_label = "LOW"

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace(0, 0), DummyFace(1, 1)]

    x_values, y_values = build_xy(DummyModel())

    assert len(x_values) == 2
    assert len(y_values) == 2
    assert y_values == [0, 1]
    assert len(x_values[0]) == 11


def test_build_xy_requires_density_id():
    class DummyFace:
        def __init__(self):
            self.features = {
                "face_id": 0,
                "surface_type_id": 0,
            }
            self.complexity_score = 0.1

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace()]

    with pytest.raises(AttributeError, match="density_id"):
        build_xy(DummyModel())


def test_attach_feature_vectors_to_model():
    class DummyFace:
        def __init__(self, face_id):
            self.features = {
                "face_id": face_id,
                "surface_type_id": 0,
                "area": 1.0,
                "perimeter": 4.0,
                "compactness": 0.8,
                "edge_count": 4,
                "neighbor_count": 2,
                "is_analytical": True,
                "is_freeform": False,
                "is_curved": False,
                "is_high_complexity_surface": False,
            }
            self.complexity_score = 0.2
            self.density_id = 0
            self.density_label = "LOW"

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace(0), DummyFace(1)]

    model = DummyModel()
    returned = attach_feature_vectors_to_model(model)

    assert returned is model
    assert hasattr(model.faces[0], "feature_vector")
    assert model.faces[0].feature_vector.face_id == 0
    assert len(model.faces[0].feature_vector.x) == 11


def test_attach_feature_vectors_to_model_rejects_none_model():
    with pytest.raises(ValueError, match="CADModel cannot be None"):
        attach_feature_vectors_to_model(None)


def test_attach_feature_vectors_to_model_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    with pytest.raises(AttributeError, match="faces"):
        attach_feature_vectors_to_model(DummyModelWithoutFaces())