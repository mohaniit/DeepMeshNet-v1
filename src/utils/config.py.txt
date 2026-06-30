from dataclasses import dataclass

@dataclass(frozen=True)
class ProjectConfig:
    random_seed: int = 42
    verbose: bool = True

CONFIG = ProjectConfig()