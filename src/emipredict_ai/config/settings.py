"""Environment-aware settings with portable project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "emi_prediction_dataset.csv"


@dataclass(frozen=True)
class Settings:
    """Minimal runtime settings used as the project grows."""

    environment: str
    dataset_path: Path
    database_url: str
    mlflow_tracking_uri: str
    mlflow_artifact_root: Path
    log_level: str


def get_settings() -> Settings:
    """Build settings from environment variables without loading secrets."""

    return Settings(
        environment=os.getenv("EMIPREDICT_ENV", "development"),
        dataset_path=Path(
            os.getenv("EMIPREDICT_DATA_PATH", str(DEFAULT_DATASET_PATH))
        ),
        database_url=os.getenv(
            "EMIPREDICT_DATABASE_URL", "sqlite:///data/emipredict.db"
        ),
        mlflow_tracking_uri=os.getenv(
            "EMIPREDICT_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
        ),
        mlflow_artifact_root=Path(
            os.getenv(
                "EMIPREDICT_MLFLOW_ARTIFACT_ROOT",
                str(PROJECT_ROOT / "artifacts" / "mlflow"),
            )
        ),
        log_level=os.getenv("EMIPREDICT_LOG_LEVEL", "INFO").upper(),
    )

