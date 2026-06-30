import pytest

from src.geometry.validator import (
    ValidationResult,
    get_validation_summary,
    require_valid_cad_model,
    validate_cad_model,
)


def test_validation_result_defaults():
    result = ValidationResult(is_valid=True)

    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.summary == {}


def test_validate_none_model_is_invalid():
    result = validate_cad_model(None)

    assert result.is_valid is False
    assert "CADModel cannot be None." in result.errors


def test_validate_model_missing_faces_and_edges():
    class DummyModel:
        pass

    result = validate_cad_model(DummyModel())

    assert result.is_valid is False
    assert "CADModel is missing faces." in result.errors
    assert "CADModel is missing edges." in result.errors
    assert "CADModel is missing vertices." in result.warnings


def test_validate_empty_model_is_invalid():
    class DummyModel:
        def __init__(self):
            self.faces = []
            self.edges = []
            self.vertices = []

    result = validate_cad_model(DummyModel())

    assert result.is_valid is False
    assert "CADModel contains no faces." in result.errors
    assert "CADModel contains no edges." in result.warnings
    assert result.summary["face_count"] == 0
    assert result.summary["edge_count"] == 0
    assert result.summary["vertex_count"] == 0


def test_validate_valid_minimal_model():
    class DummyFace:
        def __init__(self):
            self.shape = object()
            self.surface_type = "PLANE"
            self.area = 10.0
            self.perimeter = 14.0

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace()]
            self.edges = [object()]
            self.vertices = [object(), object()]

    result = validate_cad_model(DummyModel())

    assert result.is_valid is True
    assert result.errors == []
    assert result.summary["face_count"] == 1
    assert result.summary["edge_count"] == 1
    assert result.summary["vertex_count"] == 2
    assert result.summary["unknown_surface_count"] == 0
    assert result.summary["zero_area_count"] == 0
    assert result.summary["zero_perimeter_count"] == 0


def test_validate_model_warns_for_unknown_surface_and_zero_measurements():
    class DummyFace:
        def __init__(self):
            self.shape = None
            self.surface_type = "UNKNOWN"
            self.area = 0.0
            self.perimeter = 0.0

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace()]
            self.edges = [object()]
            self.vertices = []

    result = validate_cad_model(DummyModel())

    assert result.is_valid is True
    assert result.summary["unknown_surface_count"] == 1
    assert result.summary["zero_area_count"] == 1
    assert result.summary["zero_perimeter_count"] == 1
    assert any("unknown or missing surface type" in warning for warning in result.warnings)
    assert any("zero or negative area" in warning for warning in result.warnings)
    assert any("zero or negative perimeter" in warning for warning in result.warnings)
    assert any("Face 0 has no shape reference." in warning for warning in result.warnings)


def test_require_valid_cad_model_returns_valid_model():
    class DummyFace:
        def __init__(self):
            self.shape = object()
            self.surface_type = "PLANE"
            self.area = 1.0
            self.perimeter = 4.0

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace()]
            self.edges = [object()]
            self.vertices = []

    model = DummyModel()

    assert require_valid_cad_model(model) is model


def test_require_valid_cad_model_raises_for_invalid_model():
    with pytest.raises(ValueError, match="Invalid CADModel"):
        require_valid_cad_model(None)


def test_get_validation_summary():
    class DummyFace:
        def __init__(self):
            self.shape = object()
            self.surface_type = "PLANE"
            self.area = 5.0
            self.perimeter = 8.0

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace()]
            self.edges = [object(), object()]
            self.vertices = [object()]

    summary = get_validation_summary(DummyModel())

    assert summary["face_count"] == 1
    assert summary["edge_count"] == 2
    assert summary["vertex_count"] == 1