from pathlib import Path

import pytest

from src.geometry.cad_loader import load_step_file


def test_load_step_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_step_file("missing_file.step")


def test_load_step_invalid_extension_raises_error(tmp_path: Path):
    fake_file = tmp_path / "part.txt"
    fake_file.write_text("not a cad file")

    with pytest.raises(ValueError):
        load_step_file(fake_file)