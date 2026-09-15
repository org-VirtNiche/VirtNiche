from . import _utils as ul
from ._model import VirtNiche
from ._module import  VirtNicheModule

__all__ = ["VirtNiche", "VirtNicheModule"]


def _get_version() -> str:
    import importlib.metadata as importlib_metadata

    try:
        return importlib_metadata.version("VirtNiche")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


def _setup_logger() -> "logging.Logger":  
    import logging
    from importlib.metadata import version

    from rich.console import Console
    from rich.logging import RichHandler

    logger = logging.getLogger(__name__)
    # set the logging level
    logger.setLevel(logging.INFO)

    # nice logging outputs
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=False)
    formatter = logging.Formatter("VirtNiche: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # this prevents double outputs
    logger.propagate = False
    return logger


__version__ = _get_version()
logger = _setup_logger()

del _get_version, _setup_logger
