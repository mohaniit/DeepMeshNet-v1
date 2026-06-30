import math

import pytest

from src.knowledge.complexity import (
    attach_complexity_to_model,
    clamp01,
    compute_area_complexity,
    compute_compactness_complexity,
    compute_curvature_complexity,
    compute_edge_complexity,
    compute_face_complexity,
    compute_neighbor_complexity,
    compute_surface_complexity,
    get_complexity_components,
)


def test_clamp01():
    assert clamp01(-1.0) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(2.0) == 1.0


def test_surface_complexity():
    assert math.isclose(compute_surface_complexity("PLANE"), 0.10)
    assert math.isclose(compute_surface_complexity("CYLINDER"), 0.35)
    assert math.isclose(compute_surface_complexity("CONE"), 0.35)
    assert math.isclose(compute_surface_complexity("SPHERE"), 0.55)
    assert math.isclose(compute_surface_complexity("TORUS"), 0.55)
    assert math.isclose(compute_surface_complexity("BSPLINE"), 0.75)
    assert math.isclose(compute_surface_complexity("UNKNOWN"), 0.75)


def test_compactness_complexity():
    assert compute_compactness_complexity(1.0) == 0.0
    assert compute_compactness_complexity(0.0) == 1.0
    assert compute_compactness_complexity(0.5) == 0.5


def test_edge_complexity():
    assert compute_edge_complexity(0) == 0.0
    assert compute_edge_complexity(4) == 0.5
    assert compute_edge_complexity(8) == 1.0
    assert compute_edge_complexity(20) == 1.0


def test_neighbor_complexity():
    assert compute_neighbor_complexity(0) == 0.0
    assert compute_neighbor_complexity(3) == 0.5
    assert compute_neighbor_complexity(6) == 1.0
    assert compute_neighbor_complexity(20) == 1.0


def test_area_complexity():
    assert compute_area_complexity(5.0) == 1.0
    assert compute_area_complexity(1000.0) == 0.0
    assert 0.0 <= compute_area_complexity(100.0) <= 1.0


def test_curvature_complexity():
    assert compute_curvature_complexity("PLANE") == 0.0
    assert compute_curvature_complexity("CYLINDER") == 1.0
    assert compute_curvature_complexity("BSPLINE") == 1.0


def test_compute_face_complexity():
    features = {
        "surface_type": "CYLINDER",
        "compactness": 0.60,
        "edge_count": 4,
        "neighbor_count": 3,
        "area": 20.0,
    }

    score = compute_face_complexity(features)

    assert 0.0 <= score <= 1.0


def test_compute_face_complexity_none():
    with pytest.raises(ValueError):
        compute_face_complexity(None)


def test_attach_complexity_to_model():
    class DummyFace:
        def __init__(self):
            self.features = {
                "surface_type": "PLANE",
                "compactness": 0.9,
                "edge_count": 4,
                "neighbor_count": 2,
                "area": 50.0,
            }

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace(), DummyFace()]

    model = DummyModel()

    attach_complexity_to_model(model)

    for face in model.faces:
        assert hasattr(face, "complexity_score")
        assert 0.0 <= face.complexity_score <= 1.0


def test_attach_complexity_requires_features():
    class DummyFace:
        pass

    class DummyModel:
        def __init__(self):
            self.faces = [DummyFace()]

    with pytest.raises(AttributeError):
        attach_complexity_to_model(DummyModel())


def test_attach_complexity_none_model():
    with pytest.raises(ValueError):
        attach_complexity_to_model(None)


def test_get_complexity_components():
    components = get_complexity_components()

    assert len(components) == 6

    assert "surface_complexity" in components
    assert "compactness_complexity" in components
    assert "edge_complexity" in components
    assert "neighbor_complexity" in components
    assert "area_complexity" in components
    assert "curvature_complexity" in components