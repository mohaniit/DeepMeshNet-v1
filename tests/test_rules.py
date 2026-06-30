import pytest

from src.knowledge.rules import (
    DENSITY_HIGH,
    DENSITY_LOW,
    DENSITY_MEDIUM,
    DENSITY_VERY_HIGH,
    ComplexityThresholds,
    FeatureThresholds,
    complexity_to_density,
    density_to_id,
    get_density_labels,
    get_surface_rule_summary,
    id_to_density,
    is_analytical_surface,
    is_curved_surface,
    is_freeform_surface,
    is_high_complexity_surface,
    normalize_surface_type,
)


def test_density_labels_order():
    assert get_density_labels() == [
        DENSITY_LOW,
        DENSITY_MEDIUM,
        DENSITY_HIGH,
        DENSITY_VERY_HIGH,
    ]


def test_density_to_id_and_id_to_density():
    assert density_to_id("LOW") == 0
    assert density_to_id("medium") == 1
    assert density_to_id("High") == 2
    assert density_to_id("VERY_HIGH") == 3

    assert id_to_density(0) == "LOW"
    assert id_to_density(1) == "MEDIUM"
    assert id_to_density(2) == "HIGH"
    assert id_to_density(3) == "VERY_HIGH"


def test_density_to_id_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown density label"):
        density_to_id("INVALID")


def test_id_to_density_rejects_unknown_id():
    with pytest.raises(ValueError, match="Unknown density class ID"):
        id_to_density(99)


def test_normalize_surface_type():
    assert normalize_surface_type(" plane ") == "PLANE"
    assert normalize_surface_type("Cylinder") == "CYLINDER"
    assert normalize_surface_type("") == "UNKNOWN"
    assert normalize_surface_type(None) == "UNKNOWN"


def test_surface_rule_helpers():
    assert is_analytical_surface("PLANE") is True
    assert is_analytical_surface("CYLINDER") is True
    assert is_analytical_surface("BSPLINE") is False

    assert is_freeform_surface("BSPLINE") is True
    assert is_freeform_surface("UNKNOWN") is True
    assert is_freeform_surface("PLANE") is False

    assert is_curved_surface("CYLINDER") is True
    assert is_curved_surface("SPHERE") is True
    assert is_curved_surface("PLANE") is False

    assert is_high_complexity_surface("TORUS") is True
    assert is_high_complexity_surface("BSPLINE") is True
    assert is_high_complexity_surface("PLANE") is False


def test_complexity_to_density_default_thresholds():
    assert complexity_to_density(0.00) == "LOW"
    assert complexity_to_density(0.25) == "LOW"
    assert complexity_to_density(0.26) == "MEDIUM"
    assert complexity_to_density(0.50) == "MEDIUM"
    assert complexity_to_density(0.51) == "HIGH"
    assert complexity_to_density(0.75) == "HIGH"
    assert complexity_to_density(0.76) == "VERY_HIGH"
    assert complexity_to_density(1.00) == "VERY_HIGH"


def test_complexity_to_density_clamps_out_of_range_values():
    assert complexity_to_density(-10.0) == "LOW"
    assert complexity_to_density(10.0) == "VERY_HIGH"


def test_complexity_to_density_custom_thresholds():
    thresholds = ComplexityThresholds(
        low_max=0.20,
        medium_max=0.60,
        high_max=0.90,
    )

    assert complexity_to_density(0.20, thresholds) == "LOW"
    assert complexity_to_density(0.21, thresholds) == "MEDIUM"
    assert complexity_to_density(0.60, thresholds) == "MEDIUM"
    assert complexity_to_density(0.61, thresholds) == "HIGH"
    assert complexity_to_density(0.90, thresholds) == "HIGH"
    assert complexity_to_density(0.91, thresholds) == "VERY_HIGH"


def test_threshold_dataclasses_defaults():
    complexity_thresholds = ComplexityThresholds()
    feature_thresholds = FeatureThresholds()

    assert complexity_thresholds.low_max == 0.25
    assert complexity_thresholds.medium_max == 0.50
    assert complexity_thresholds.high_max == 0.75

    assert feature_thresholds.small_area == 10.0
    assert feature_thresholds.large_area == 1000.0
    assert feature_thresholds.low_compactness == 0.25
    assert feature_thresholds.high_edge_count == 8
    assert feature_thresholds.high_neighbor_count == 6


def test_get_surface_rule_summary():
    summary = get_surface_rule_summary()

    assert "analytical_surfaces" in summary
    assert "freeform_surfaces" in summary
    assert "curved_surfaces" in summary
    assert "high_complexity_surfaces" in summary

    assert "PLANE" in summary["analytical_surfaces"]
    assert "BSPLINE" in summary["freeform_surfaces"]
    assert "CYLINDER" in summary["curved_surfaces"]
    assert "TORUS" in summary["high_complexity_surfaces"]