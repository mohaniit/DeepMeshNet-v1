from src.utils.banner import print_banner
from src.utils.logger import get_logger
from src.utils.paths import ROOT

print_banner("Infrastructure Test")

logger = get_logger()

logger.info("DeepMeshNet-v1 initialized successfully.")
logger.info(f"Project root: {ROOT}")