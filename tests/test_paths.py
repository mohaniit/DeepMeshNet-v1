from src.utils.paths import (
    ROOT,
    DATA_DIR,
    DOCS_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    CM500_DIR,
)


def test_project_root_exists():
    assert ROOT.exists()
    assert ROOT.name == "DeepMeshNet-v1"


def test_main_directories_exist():
    assert DATA_DIR.exists()
    assert DOCS_DIR.exists()
    assert OUTPUT_DIR.exists()
    assert CM500_DIR.exists()


def test_output_subdirectories_exist():
    assert LOG_DIR.exists()
    assert FIGURE_DIR.exists()
    assert MODEL_DIR.exists()