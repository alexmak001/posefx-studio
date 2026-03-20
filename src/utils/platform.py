"""Platform detection for cross-device compatibility."""

import logging

logger = logging.getLogger(__name__)


def detect_platform() -> str:
    """Detect the current hardware platform.

    Returns:
        'mac' if MPS is available (Apple Silicon),
        'jetson' if CUDA is available,
        'cpu' otherwise.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            logger.info("Platform detected: mac (MPS available)")
            return "mac"
        if torch.cuda.is_available():
            logger.info("Platform detected: jetson (CUDA available)")
            return "jetson"
    except ImportError:
        logger.warning("torch not available, falling back to CPU platform detection")

    logger.info("Platform detected: cpu")
    return "cpu"
