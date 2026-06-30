from src.geometry.backend import (
    active_cad_backend,
    cad_backend_available,
    get_backend_info,
    require_cad_backend,
)


def test_backend_info_is_consistent():
    info = get_backend_info()

    assert info.name in {"OCP", "unavailable"}
    assert info.available in {True, False}
    assert active_cad_backend() == info.name
    assert cad_backend_available() == info.available


def test_require_backend_behavior():
    if cad_backend_available():
        require_cad_backend()
    else:
        try:
            require_cad_backend()
        except ImportError as exc:
            assert "OpenCascade/OCP backend is unavailable" in str(exc)