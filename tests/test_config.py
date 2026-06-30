from src.utils.config import CONFIG


def test_config_defaults():
    assert CONFIG.random_seed == 42
    assert CONFIG.verbose is True