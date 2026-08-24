"""Phase 0 smoke tests for package and settings imports."""

from emipredict_ai import __version__
from emipredict_ai.config import get_settings


def test_package_version() -> None:
    """The installable project package exposes its initial version."""

    assert __version__ == "0.1.0"


def test_default_dataset_location() -> None:
    """The default configuration points at the Git-ignored raw data location."""

    settings = get_settings()

    assert settings.dataset_path.name == "emi_prediction_dataset.csv"
    assert settings.dataset_path.parent.name == "raw"

