import pytest

from src.geometry.cad_model import CADModel
from src.geometry.topology import extract_topology


def test_extract_topology_requires_shape():
    model = CADModel(
        model_id="CM0001",
        file_name="CM0001.step",
    )

    with pytest.raises(ValueError):
        extract_topology(model)