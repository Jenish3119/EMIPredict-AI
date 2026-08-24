# EMIPredict AI — Intelligent Financial Risk Assessment Platform

An end-to-end FinTech machine learning platform for EMI eligibility classification
and maximum affordable EMI prediction using MLflow and Streamlit.

## 🔗 Project Links

- **Live Demo:** https://emipredict-ai-6gzynwugtr7pzemxek7lvo.streamlit.app/
- **GitHub Repository:** https://github.com/Jenish3119/EMIPredict-AI

## Project Overview

EMIPredict AI is a planned decision-support platform for financial-risk and EMI-affordability assessment. It will combine reusable data and machine-learning pipelines with MLflow tracking, explainability, a FastAPI backend, SQL persistence, and a Streamlit user interface.

> Status: Phase 0 - project foundation. No dataset analysis, model training, or performance results have been produced yet.

## Planned machine-learning problems

1. Multiclass classification of `emi_eligibility` into `Eligible`, `High_Risk`, or `Not_Eligible` after the real target labels are verified.
2. Regression of `max_monthly_emi`, representing a model estimate of a customer's maximum safe monthly EMI after the real target is verified.

The platform is intended for educational and decision-support use. It is not an automated lending authority and does not claim regulatory compliance.

## Planned architecture

```text
Raw data -> Validation -> Cleaning -> EDA -> Feature engineering
         -> Classification + Regression -> MLflow registry
         -> Explainability + Recommendations -> FastAPI
         -> SQL database -> Streamlit -> Deployment
```

## Repository layout

```text
.
|-- app_pages/                   # Streamlit page scripts (implemented later)
|-- artifacts/                   # Generated models, plots, and reports
|-- data/                        # Local raw, interim, and processed data
|-- docs/                        # Technical documentation
|-- notebooks/                   # Exploration only; reusable logic stays in src
|-- scripts/                     # Command-line pipeline entry points
|-- src/
|   `-- emipredict_ai/           # Installable application package
|       |-- api/
|       |-- config/
|       |-- data/
|       |-- database/
|       |-- explainability/
|       |-- features/
|       |-- models/
|       |-- recommendations/
|       `-- utils/
`-- tests/                       # Automated tests
```

The named `src/emipredict_ai` package is intentionally used instead of generic top-level packages such as `src/config`. This prevents import-name collisions and makes local, test, API, and deployment imports consistent.

The future Streamlit entry point will be `streamlit_app.py`, with navigation defined through `st.navigation` and page scripts stored in `app_pages/`. This follows the current Streamlit multi-page API and avoids the legacy `pages/` auto-discovery behavior.

## Local setup

The project currently targets Python 3.10 through 3.12. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Run the Phase 0 verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Dataset location

The dataset is deliberately excluded from Git. Before Phase 1, place the real CSV at:

```text
data/raw/emi_prediction_dataset.csv
```

The currently located source file is outside the repository at:

```text
C:\Users\jenis\Downloads\emi_prediction_dataset.csv
```

Phase 1 will inspect the file programmatically before any schema, statistics, cleaning rules, or model assumptions are made.

## Next phase

Phase 1 will perform a read-only dataset audit and generate a reusable data-quality report. It will not delete or modify source records automatically.
