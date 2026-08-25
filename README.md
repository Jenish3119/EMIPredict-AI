# EMIPredict AI

EMIPredict AI is an educational decision-support project that predicts an EMI eligibility category and estimates a maximum safe monthly EMI. The project uses one beginner-friendly Jupyter notebook for the complete technical workflow and one Python file for the multi-page Streamlit application.

GitHub repository: [Jenish3119/EMIPredict-AI](https://github.com/Jenish3119/EMIPredict-AI)

> The project is not a loan-approval system or financial advice. A qualified human and the financial institution's policies must make any real lending decision.

## Deliverable status

| Deliverable | Evidence | Status |
|---|---|---|
| Data cleaning and preprocessing | `EMIPredict_AI_Project.ipynb`, sections 2–4 | Complete |
| Feature engineering and transformations | Notebook section 5 | Complete |
| EDA and visualizations | Notebook section 6 and its saved outputs | Complete |
| Three classification models | Logistic Regression, Random Forest, XGBoost | Complete |
| Three regression models | Linear Regression, Random Forest, XGBoost | Complete |
| Best-model selection and comparison | Notebook sections 10–12 and comparison CSV files | Complete |
| MLflow tracking and registry | Two experiments and two registered selected models | Complete locally |
| Multi-page real-time Streamlit app | `streamlit_app.py` | Complete and locally tested |
| GitHub codebase and documentation | This repository and README | Complete |
| Public Streamlit Cloud URL | Deploy from this repository | Pending |

## Project files for evaluation

- `EMIPredict_AI_Project.ipynb` — all data analysis, preprocessing, feature engineering, model training, evaluation, MLflow tracking, registry use, and artifact saving.
- `streamlit_app.py` — Home, Customer assessment, Model information, and Project report pages in a single beginner-friendly Python file.
- `artifacts/models/` — the two selected fitted pipelines, metadata, and validation comparison tables required by the app.
- `requirements.txt` — minimal pinned cloud dependencies that match the saved models.

The original dataset is excluded from Git because it is large. The notebook looks first for `data/raw/emi_prediction_dataset.csv` and then for `Downloads/emi_prediction_dataset.csv`.

## Dataset and objectives

The supplied dataset contains 404,800 rows and 27 original columns. There are 25 original input columns and two targets:

1. `emi_eligibility` — multiclass classification into `Eligible`, `High_Risk`, or `Not_Eligible`.
2. `max_monthly_emi` — regression estimate of the maximum safe monthly EMI.

No duplicate rows were detected. Missingness was low: approximately 0.58%–0.60% in education, monthly rent, credit score, bank balance, and emergency fund. Cleaning also standardizes inconsistent gender labels, converts malformed numeric values, and marks out-of-range values as missing before imputation.

## Methodology and architecture

```text
Raw CSV (404,800 rows)
        |
        v
Data-quality audit and cleaning
        |
        v
11 transparent financial features
        |
        v
Stratified train / validation / test split
        |
        +--> 3 classification models --> macro-F1 selection
        |
        +--> 3 regression models -----> RMSE selection
        |
        v
MLflow experiments and model registry
        |
        v
Selected pipelines saved with preprocessing
        |
        v
Multi-page Streamlit real-time assessment
```

The 11 engineered features include total monthly expenses, annual income, expense-to-income ratio, existing EMI burden, disposable income, loan-to-income ratio, emergency-fund coverage, balance-to-income ratio, dependent burden, a requested-payment proxy, and financial cushion. Together with the original inputs, the model receives 36 features.

The preprocessing pipelines use:

- median imputation and standardization for numeric columns;
- most-frequent imputation and one-hot encoding for categorical columns;
- safe handling of previously unseen categories;
- fixed random seeds and stratified splitting for reproducibility.

## Exploratory findings and business interpretation

- Eligibility is imbalanced: 77.29% Not Eligible, 18.39% Eligible, and 4.32% High Risk. This is why macro F1 is the primary classification-selection metric rather than accuracy alone.
- Profiles without an existing loan were Eligible in 25.78% of rows, compared with 7.27% for profiles with an existing loan. This is an association in the supplied dataset and does not prove causation.
- Vehicle EMI and Personal Loan EMI had the highest Not Eligible proportions: 86.15% and 85.25%.
- Median maximum monthly EMI was ₹13,840 for Eligible profiles, ₹10,285 for High Risk profiles, and ₹2,464 for Not Eligible profiles.
- The eligibility-by-scenario and salary-versus-maximum-EMI charts are displayed directly in the executed notebook.

## Model comparison and selection

### Classification validation results

| Model | Accuracy | Macro F1 | ROC AUC OVR |
|---|---:|---:|---:|
| XGBoost Classifier | 96.33% | 79.78% | 99.06% |
| Random Forest Classifier | 94.29% | 78.24% | 97.89% |
| Logistic Regression | 91.24% | 60.33% | 95.08% |

XGBoost was selected because it produced the highest validation macro F1. On the untouched test set it achieved 96.38% accuracy, 80.15% macro F1, and 99.05% one-vs-rest ROC AUC.

### Regression validation results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost Regressor | ₹312.32 | ₹854.05 | 98.79% |
| Random Forest Regressor | ₹239.51 | ₹921.21 | 98.60% |
| Linear Regression | ₹2,899.80 | ₹4,063.09 | 72.71% |

XGBoost was selected using the lowest validation RMSE, the declared selection metric. Random Forest produced a lower MAE, which is reported transparently. On the untouched test set, XGBoost achieved MAE ₹310.96, RMSE ₹851.12, and R² 98.79%.

The saved run used `FAST_MODE = True`: 100,000 stratified training rows and the complete 60,720-row validation and 60,720-row test sets. The quality audit and EDA use all 404,800 rows. Set `FAST_MODE = False` in the notebook when full training is required and execution time is available.

## MLflow evidence

The notebook logs each model variant to separate `EMIPredict_Classification` and `EMIPredict_Regression` experiments. It records model names, fast-mode status, validation metrics, and model artifacts. The selected models are registered as:

- `EMIPredict_Eligibility_Model`
- `EMIPredict_Max_EMI_Model`

To inspect the local experiment dashboard after running the notebook:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`.

## Business impact and recommendations

The application can support a consistent first-pass affordability discussion, prioritize High Risk applications for manual review, and show customers how requested amount, tenure, expenses, and existing EMI burden affect affordability indicators.

For a financial institution:

- keep a human decision-maker and apply internal lending policies after the model result;
- use High Risk as a review flag rather than an automatic rejection;
- monitor class balance, macro F1, regression error, and prediction drift;
- validate fairness across legally relevant groups before real use;
- retrain and independently validate the model when customer behaviour or policy changes;
- protect applicant data and keep prediction logs under appropriate access controls.

## Local setup and use

Python 3.10 is recommended because it matches the saved model environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open and run `EMIPredict_AI_Project.ipynb` from top to bottom if the models need to be retrained. To use the already saved models, start the app directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Run the automated project checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Streamlit Community Cloud deployment

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) using the GitHub account that can access the repository.
2. Choose **Create app** and select `Jenish3119/EMIPredict-AI`.
3. Select branch `main` and entry point `streamlit_app.py`.
4. In advanced settings, choose Python 3.10 if a version choice is shown.
5. Deploy the app. The tracked model artifacts let it make predictions without the original CSV or local MLflow database.
6. Add the resulting public URL near the top of this README.

## Limitations and responsible use

- The supplied dataset may be synthetic or otherwise unrepresentative of real applicants.
- The model has not completed production fairness, calibration, stress, security, or regulatory validation.
- The requested-payment proxy is requested amount divided by tenure; it is not a bank EMI calculation because no interest rate is supplied.
- A high model score does not establish creditworthiness, eligibility under law, or final approval.
- Predictions must remain educational decision support unless a financial institution completes independent governance and validation.
