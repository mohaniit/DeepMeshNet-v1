import pytest

from src.learning.model_registry import (
    ModelRegistry,
    clear_global_registry,
    create_model,
    get_global_registry,
    has_model,
    register_model,
    registered_models,
)


class DummyModel:
    def __init__(self, value=0):
        self.value = value


def dummy_factory(**kwargs):
    return DummyModel(**kwargs)


def test_model_registry_initialization():
    registry = ModelRegistry()

    assert registry.names() == []
    assert registry.summary() == {
        "num_models": 0,
        "models": [],
    }


def test_register_model():
    registry = ModelRegistry()

    registry.register("dummy", dummy_factory)

    assert registry.has_model("dummy") is True
    assert registry.names() == ["dummy"]


def test_register_model_normalizes_name():
    registry = ModelRegistry()

    registry.register("  Dummy  ", dummy_factory)

    assert registry.has_model("dummy") is True
    assert registry.has_model("DUMMY") is True


def test_register_duplicate_model_raises():
    registry = ModelRegistry()

    registry.register("dummy", dummy_factory)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", dummy_factory)


def test_register_duplicate_model_with_overwrite():
    registry = ModelRegistry()

    registry.register("dummy", dummy_factory)
    registry.register("dummy", dummy_factory, overwrite=True)

    assert registry.has_model("dummy") is True


def test_register_rejects_empty_name():
    registry = ModelRegistry()

    with pytest.raises(ValueError, match="empty"):
        registry.register("", dummy_factory)


def test_register_rejects_none_name():
    registry = ModelRegistry()

    with pytest.raises(ValueError, match="cannot be None"):
        registry.register(None, dummy_factory)


def test_register_rejects_none_factory():
    registry = ModelRegistry()

    with pytest.raises(ValueError, match="factory cannot be None"):
        registry.register("dummy", None)


def test_create_model():
    registry = ModelRegistry()

    registry.register("dummy", dummy_factory)

    model = registry.create("dummy", value=42)

    assert isinstance(model, DummyModel)
    assert model.value == 42


def test_create_missing_model_raises():
    registry = ModelRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.create("missing")


def test_get_factory():
    registry = ModelRegistry()

    registry.register("dummy", dummy_factory)

    assert registry.get_factory("dummy") is dummy_factory


def test_get_missing_factory_raises():
    registry = ModelRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get_factory("missing")


def test_clear_registry():
    registry = ModelRegistry()

    registry.register("dummy", dummy_factory)
    registry.clear()

    assert registry.names() == []


def test_registry_summary():
    registry = ModelRegistry()

    registry.register("b", dummy_factory)
    registry.register("a", dummy_factory)

    assert registry.summary() == {
        "num_models": 2,
        "models": ["a", "b"],
    }


def test_global_registry_register_and_create():
    clear_global_registry()

    register_model("dummy", dummy_factory)

    model = create_model("dummy", value=7)

    assert isinstance(model, DummyModel)
    assert model.value == 7
    assert has_model("dummy") is True
    assert registered_models() == ["dummy"]

    clear_global_registry()


def test_get_global_registry():
    registry = get_global_registry()

    assert isinstance(registry, ModelRegistry)


def test_clear_global_registry():
    clear_global_registry()

    register_model("dummy", dummy_factory)

    assert has_model("dummy") is True

    clear_global_registry()

    assert registered_models() == []