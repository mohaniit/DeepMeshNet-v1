import pytest

from src.knowledge.density_labels import (
    assign_density_id,
    assign_density_label,
    attach_density_labels_to_model,
    get_density_label_fields,
    safe_complexity_score,
    summarize_density_labels,
)


def test_safe_complexity_score():
    assert safe_complexity_score(0.5) == 0.5
    assert safe_complexity_score(-1.0) == 0.0
    assert safe_complexity_score(2.0) == 1.0
    assert safe_complexity_score(None) == 0.0
    assert safe_complexity_score("bad") == 0.0
    assert safe_complexity_score(float("nan")) == 0.0


def test_assign_density_label():
    assert assign_density_label(0.00) == "LOW"
    assert assign_density_label(0.25) == "LOW"
    assert assign_density_label(0.26) == "MEDIUM"
    assert assign_density_label(0.50) == "MEDIUM"
    assert assign_density_label(0.51) == "HIGH"
    assert assign_density_label(0.75) == "HIGH"
    assert assign_density_label(0.76) == "VERY_HIGH"


def test_assign_density_id():
    assert assign_density_id(0.10) == 0
    assert assign_density_id(0.40) == 1
    assert assign_density_id(0.60) == 2
    assert assign_density_id(0.90) == 3


def test_attach_density_labels_to_model():
    class DummyFace:
        def __init__(self, score):
            self.complexity_score = score

    class DummyModel:
        def __init__(self):
            self.faces = [
                DummyFace(0.10),
                DummyFace(0.40),
                DummyFace(0.60),
                DummyFace(0.90),
            ]

    model = DummyModel()
    returned = attach_density_labels_to_model(model)

    assert returned is model

    assert model.faces[0].density_label == "LOW"
    assert model.faces[0].density_id == 0

    assert model.faces[1].density_label == "MEDIUM"
    assert model.faces[1].density_id == 1

    assert model.faces[2].density_label == "HIGH"
    assert model.faces[2].density_id == 2

    assert model.faces[3].density_label == "VERY_HIGH"
    assert model.faces[3].density_id == 3


def test_attach_density_labels_to_model_rejects_none_model():
    with pytest.raises(ValueError, match="CADModel cannot be None"):
        attach_density_labels_to_model(None)


def test_attach_density_labels_to_model_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    with pytest.raises(AttributeError, match="faces"):
        attach_density_labels_to_model(DummyModelWithoutFaces())


def test_attach_density_labels_to_model_requires_complexity_score():
    class DummyFace:
        pass

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace()]

    with pytest.raises(AttributeError, match="complexity_score"):
        attach_density_labels_to_model(DummyModel())


def test_summarize_density_labels():
    class DummyFace:
        def __init__(self, label):
            self.density_label = label

    class DummyModel:
        def __init__(self):
            self.faces = [
                DummyFace("LOW"),
                DummyFace("LOW"),
                DummyFace("MEDIUM"),
                DummyFace("HIGH"),
                DummyFace("VERY_HIGH"),
                DummyFace("UNKNOWN"),
            ]

    summary = summarize_density_labels(DummyModel())

    assert summary == {
        "LOW": 2,
        "MEDIUM": 1,
        "HIGH": 1,
        "VERY_HIGH": 1,
    }


def test_summarize_density_labels_rejects_none_model():
    with pytest.raises(ValueError, match="CADModel cannot be None"):
        summarize_density_labels(None)


def test_summarize_density_labels_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    with pytest.raises(AttributeError, match="faces"):
        summarize_density_labels(DummyModelWithoutFaces())


def test_get_density_label_fields():
    assert get_density_label_fields() == [
        "density_label",
        "density_id",
    ]