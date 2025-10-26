# llmextract/__init__.py

import logging
from importlib.metadata import PackageNotFoundError, version

logging.getLogger(__name__).addHandler(logging.NullHandler())


try:
    __version__ = version("llmextract")
except PackageNotFoundError:
    __version__ = "0.0.0"


def configure_logging(verbosity: int = 0) -> None:
    """
    Configure llmextract package-level logging.

    verbosity:
      0 - minimal (WARNING)
      1 - INFO
      2 - DEBUG
    """
    pkg_logger = logging.getLogger("llmextract")
    if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        pkg_logger.addHandler(handler)

    if verbosity >= 2:
        pkg_logger.setLevel(logging.DEBUG)
    elif verbosity >= 1:
        pkg_logger.setLevel(logging.INFO)
    else:
        pkg_logger.setLevel(logging.WARNING)


from .data_models import AnnotatedDocument, CharInterval, ExampleData, Extraction  # noqa: E402
from .services import extract, aextract  # noqa: E402
from .visualization import visualize  # noqa: E402

__all__ = [
    "extract",
    "aextract",
    "visualize",
    "AnnotatedDocument",
    "CharInterval",
    "ExampleData",
    "Extraction",
    "configure_logging",
    "__version__",
]
