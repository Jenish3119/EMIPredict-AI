"""Single-file Streamlit app for the EMIPredict AI notebook artifacts.

Run the notebook first. It creates the model files in artifacts/models/.
Then start this application with:
    .\\.venv\\Scripts\\streamlit.exe run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
CLASSIFICATION_MODEL_PATH = MODEL_DIR / "best_classification_pipeline.joblib"
REGRESSION_MODEL_PATH = MODEL_DIR / "best_regression_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
CLASSIFICATION_RESULTS_PATH = MODEL_DIR / "classification_comparison.csv"
REGRESSION_RESULTS_PATH = MODEL_DIR / "regression_comparison.csv"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely and keep an undefined ratio as missing instead of infinity."""

    denominator = denominator.mask(denominator == 0)
    return numerator / denominator


def add_financial_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the same financial features used by the training notebook."""

    result = frame.copy()
    expense_columns = [
        "monthly_rent",
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
    ]
    result["total_monthly_expenses"] = result[expense_columns].sum(axis=1)
    result["annual_income"] = result["monthly_salary"] * 12
    result["expense_to_income_ratio"] = safe_divide(
        result["total_monthly_expenses"], result["monthly_salary"]
    )
    result["existing_emi_burden"] = safe_divide(
        result["current_emi_amount"], result["monthly_salary"]
    )
    result["disposable_income"] = (
        result["monthly_salary"]
        - result["total_monthly_expenses"]
        - result["current_emi_amount"]
    )
    result["loan_to_income_ratio"] = safe_divide(
        result["requested_amount"], result["annual_income"]
    )
    result["emergency_fund_coverage"] = safe_divide(
        result["emergency_fund"], result["total_monthly_expenses"]
    )
    result["balance_to_income_ratio"] = safe_divide(
        result["bank_balance"], result["monthly_salary"]
    )
    result["dependent_burden"] = safe_divide(
        result["dependents"], result["family_size"]
    )
    result["requested_payment_proxy"] = safe_divide(
        result["requested_amount"], result["requested_tenure"]
    )
    result["financial_cushion"] = (
        result["disposable_income"] - result["requested_payment_proxy"]
    )
    return result


@st.cache_resource
def load_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    """Load fitted pipelines once per Streamlit process."""

    required_files = [
        CLASSIFICATION_MODEL_PATH,
        REGRESSION_MODEL_PATH,
        METADATA_PATH,
    ]
    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run EMIPredict_AI_Project.ipynb first."
        )

    classifier = joblib.load(CLASSIFICATION_MODEL_PATH)
    regressor = joblib.load(REGRESSION_MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return classifier, regressor, metadata


def format_inr(amount: float) -> str:
    """Display a number as an INR amount without claiming bank approval."""

    return f"₹{amount:,.0f}"


def build_recommendation(
    eligibility: str,
    predicted_emi: float,
    engineered_input: pd.DataFrame,
) -> list[str]:
    """Return transparent, educational observations from calculated ratios."""

    row = engineered_input.iloc[0]
    notes: list[str] = []

    if row["existing_emi_burden"] > 0.30:
        notes.append("Existing EMI burden is high relative to monthly salary.")
    if row["expense_to_income_ratio"] > 0.50:
        notes.append("Monthly living expenses take a large share of income.")
    if row["emergency_fund_coverage"] < 3:
        notes.append("Emergency savings cover fewer than three months of expenses.")
    if row["financial_cushion"] < 0:
        notes.append("The simple requested-payment proxy exceeds disposable income.")
    if row["credit_score"] < 650:
        notes.append("The entered credit score is relatively low in this dataset's range.")

    if not notes:
        notes.append("No simple ratio-based warning was triggered for this entered profile.")

    if eligibility == "Not_Eligible":
        notes.append(
            "Consider reducing the requested amount or reviewing the repayment tenure."
        )
    elif eligibility == "High_Risk":
        notes.append(
            "Treat this result as a caution signal and review the full financial profile."
        )
    else:
        notes.append(
            "The model result is favourable, but it is still decision support, not approval."
        )

    notes.append(f"The model-estimated maximum safe EMI is {format_inr(predicted_emi)} per month.")
    return notes


def render_home() -> None:
    """Show a short, evaluation-friendly overview."""

    st.title("EMIPredict AI")
    st.caption("Intelligent financial-risk and EMI-affordability assessment")

    with st.container(border=True):
        st.subheader("What this app does")
        st.write(
            "It uses the two models trained in the notebook to estimate an EMI "
            "eligibility category and a maximum safe monthly EMI amount."
        )

    with st.container(border=True):
        st.subheader("Project workflow")
        st.write(
            "The notebook cleans the 404,800-row dataset, engineers transparent "
            "financial features, compares three classification and three regression "
            "models, records the experiments with MLflow, and saves the selected "
            "pipelines used by this application."
        )

    st.info(
        "This is an educational decision-support project. It is not an automated "
        "loan-approval system and does not replace a lender's policy or human review."
    )


def render_assessment() -> None:
    """Collect one customer profile and display model predictions."""

    st.title("Customer assessment")
    st.caption("Enter a sample financial profile, then submit once to run both models.")

    try:
        classifier, regressor, metadata = load_artifacts()
    except FileNotFoundError as error:
        st.warning(str(error))
        return

    with st.form("customer_assessment", border=True):
        st.subheader("Personal and employment details")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Female", "Male"])
            marital_status = st.selectbox("Marital status", ["Single", "Married"])
            education = st.selectbox(
                "Education",
                ["High School", "Graduate", "Post Graduate", "Professional"],
            )
        with col2:
            monthly_salary = st.number_input(
                "Monthly salary (₹)", min_value=1_000.0, value=60_000.0, step=1_000.0
            )
            employment_type = st.selectbox(
                "Employment type", ["Private", "Government", "Self-employed"]
            )
            years_of_employment = st.number_input(
                "Years of employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5
            )
            company_type = st.selectbox(
                "Company type", ["Small", "Startup", "Mid-size", "Large Indian", "MNC"]
            )
        with col3:
            house_type = st.selectbox("House type", ["Rented", "Own", "Family"])
            family_size = st.number_input("Family size", min_value=1, max_value=20, value=3)
            dependents = st.number_input(
                "Dependents", min_value=0, max_value=int(family_size), value=1
            )
            credit_score = st.number_input(
                "Credit score", min_value=300.0, max_value=850.0, value=700.0, step=1.0
            )

        st.subheader("Monthly obligations and financial status")
        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_rent = st.number_input("Monthly rent (₹)", min_value=0.0, value=15_000.0, step=500.0)
            school_fees = st.number_input("School fees (₹)", min_value=0.0, value=0.0, step=500.0)
            college_fees = st.number_input("College fees (₹)", min_value=0.0, value=0.0, step=500.0)
        with col2:
            travel_expenses = st.number_input("Travel expenses (₹)", min_value=0.0, value=4_000.0, step=500.0)
            groceries_utilities = st.number_input("Groceries and utilities (₹)", min_value=0.0, value=12_000.0, step=500.0)
            other_monthly_expenses = st.number_input("Other monthly expenses (₹)", min_value=0.0, value=5_000.0, step=500.0)
        with col3:
            existing_loans = st.selectbox("Existing loans", ["No", "Yes"])
            current_emi_amount = st.number_input("Current EMI amount (₹)", min_value=0.0, value=0.0, step=500.0)
            bank_balance = st.number_input("Bank balance (₹)", min_value=0.0, value=150_000.0, step=1_000.0)
            emergency_fund = st.number_input("Emergency fund (₹)", min_value=0.0, value=75_000.0, step=1_000.0)

        st.subheader("Requested EMI details")
        col1, col2, col3 = st.columns(3)
        with col1:
            emi_scenario = st.selectbox(
                "EMI scenario",
                [
                    "E-commerce Shopping EMI",
                    "Home Appliances EMI",
                    "Vehicle EMI",
                    "Personal Loan EMI",
                    "Education EMI",
                ],
            )
        with col2:
            requested_amount = st.number_input(
                "Requested amount (₹)", min_value=1_000.0, value=250_000.0, step=1_000.0
            )
        with col3:
            requested_tenure = st.number_input(
                "Requested tenure (months)", min_value=1, max_value=120, value=24
            )

        submitted = st.form_submit_button(
            "Run assessment", type="primary", icon=":material/analytics:"
        )

    if not submitted:
        return

    raw_input = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "marital_status": marital_status,
                "education": education,
                "monthly_salary": monthly_salary,
                "employment_type": employment_type,
                "years_of_employment": years_of_employment,
                "company_type": company_type,
                "house_type": house_type,
                "monthly_rent": monthly_rent,
                "family_size": family_size,
                "dependents": dependents,
                "school_fees": school_fees,
                "college_fees": college_fees,
                "travel_expenses": travel_expenses,
                "groceries_utilities": groceries_utilities,
                "other_monthly_expenses": other_monthly_expenses,
                "existing_loans": existing_loans,
                "current_emi_amount": current_emi_amount,
                "credit_score": credit_score,
                "bank_balance": bank_balance,
                "emergency_fund": emergency_fund,
                "emi_scenario": emi_scenario,
                "requested_amount": requested_amount,
                "requested_tenure": requested_tenure,
            }
        ]
    )
    engineered_input = add_financial_features(raw_input)
    model_input = engineered_input.reindex(columns=metadata["feature_columns"])

    predicted_code = int(classifier.predict(model_input)[0])
    class_labels = metadata["class_labels"]
    eligibility = class_labels[predicted_code]
    probabilities = classifier.predict_proba(model_input)[0]
    predicted_emi = float(regressor.predict(model_input)[0])

    st.subheader("Assessment result")
    with st.container(horizontal=True):
        st.metric("Eligibility", eligibility, border=True)
        st.metric("Maximum recommended EMI", format_inr(predicted_emi), border=True)
        st.metric(
            "Disposable income",
            format_inr(float(engineered_input.loc[0, "disposable_income"])),
            border=True,
        )

    probability_table = pd.DataFrame(
        {"Eligibility class": class_labels, "Probability": probabilities}
    )
    with st.container(border=True):
        st.subheader("Eligibility probabilities")
        st.bar_chart(probability_table, x="Eligibility class", y="Probability")
        st.dataframe(
            probability_table,
            column_config={
                "Probability": st.column_config.NumberColumn(format="percent")
            },
            hide_index=True,
        )

    ratio_table = pd.DataFrame(
        {
            "Measure": [
                "Expense-to-income ratio",
                "Existing EMI burden",
                "Emergency-fund coverage (months)",
                "Requested payment proxy",
                "Financial cushion",
            ],
            "Value": [
                f"{engineered_input.loc[0, 'expense_to_income_ratio']:.1%}",
                f"{engineered_input.loc[0, 'existing_emi_burden']:.1%}",
                f"{engineered_input.loc[0, 'emergency_fund_coverage']:.1f}",
                format_inr(float(engineered_input.loc[0, "requested_payment_proxy"])),
                format_inr(float(engineered_input.loc[0, "financial_cushion"])),
            ],
        }
    )
    with st.container(border=True):
        st.subheader("Calculated financial measures")
        st.dataframe(ratio_table, hide_index=True)

    with st.container(border=True):
        st.subheader("Transparent recommendation notes")
        for note in build_recommendation(eligibility, predicted_emi, engineered_input):
            st.write(f"- {note}")

    st.caption(
        "The requested-payment value is only requested amount ÷ tenure. It is not a "
        "bank EMI calculation because no interest rate is entered."
    )


def render_model_information() -> None:
    """Show real results saved by the notebook without inventing metrics."""

    st.title("Model information")
    try:
        _, _, metadata = load_artifacts()
    except FileNotFoundError as error:
        st.warning(str(error))
        return

    with st.container(border=True):
        st.subheader("Selected models")
        st.write(f"Classification: `{metadata['best_classification_model']}`")
        st.write(f"Regression: `{metadata['best_regression_model']}`")
        st.caption(
            "Classification is selected by validation macro F1. Regression is "
            "selected by validation RMSE; lower RMSE is better."
        )

    classification_metrics = metadata["classification_test_metrics"]
    regression_metrics = metadata["regression_test_metrics"]

    st.subheader("Untouched test-set performance")
    with st.container(horizontal=True):
        st.metric(
            "Classification accuracy",
            f"{classification_metrics['accuracy']:.2%}",
            border=True,
        )
        st.metric(
            "Classification macro F1",
            f"{classification_metrics['macro_f1']:.2%}",
            border=True,
        )
        st.metric(
            "Classification ROC AUC",
            f"{classification_metrics['roc_auc_ovr']:.2%}",
            border=True,
        )

    with st.container(horizontal=True):
        st.metric(
            "Regression MAE",
            format_inr(regression_metrics["mae"]),
            border=True,
        )
        st.metric(
            "Regression RMSE",
            format_inr(regression_metrics["rmse"]),
            border=True,
        )
        st.metric(
            "Regression R²",
            f"{regression_metrics['r2']:.2%}",
            border=True,
        )

    if CLASSIFICATION_RESULTS_PATH.exists():
        with st.container(border=True):
            st.subheader("Classification comparison")
            st.dataframe(pd.read_csv(CLASSIFICATION_RESULTS_PATH), hide_index=True)

    if REGRESSION_RESULTS_PATH.exists():
        with st.container(border=True):
            st.subheader("Regression comparison")
            st.dataframe(pd.read_csv(REGRESSION_RESULTS_PATH), hide_index=True)

    if metadata.get("fast_mode"):
        st.warning(
            "The saved training run used the notebook's beginner-friendly fast mode: "
            "100,000 stratified training rows plus the complete 60,720-row validation "
            "and 60,720-row test sets. The data-quality audit used all 404,800 rows."
        )

    st.info(
        "MLflow records every model variant in the local classification and regression "
        "experiments. The selected pipelines are registered as "
        "`EMIPredict_Eligibility_Model` and `EMIPredict_Max_EMI_Model`."
    )


def render_project_report() -> None:
    """Summarize methodology, evidence, business impact, and limitations."""

    st.title("Project report")
    st.caption("Methodology, exploratory insights, and responsible-use recommendations")

    with st.container(border=True):
        st.subheader("Methodology and architecture")
        st.code(
            "Raw CSV (404,800 rows)\n"
            "  → quality checks and cleaning\n"
            "  → 11 financial features\n"
            "  → train / validation / test split\n"
            "  → 3 classifiers + 3 regressors\n"
            "  → MLflow comparison and registry\n"
            "  → selected pipelines\n"
            "  → Streamlit real-time assessment",
            language=None,
        )
        st.write(
            "Numeric missing values are imputed with the median and scaled. "
            "Categorical missing values use the most frequent category and are "
            "one-hot encoded. Unknown categories are handled safely."
        )

    st.subheader("Data-quality and exploratory findings")
    with st.container(horizontal=True):
        st.metric("Dataset rows", "404,800", border=True)
        st.metric("Original columns", "27", border=True)
        st.metric("Engineered model inputs", "36", border=True)

    with st.container(border=True):
        st.markdown(
            "- No duplicate rows were detected.\n"
            "- Missingness was low—about 0.58% to 0.60% in education, rent, "
            "credit score, bank balance, and emergency fund.\n"
            "- Eligibility was imbalanced: 77.29% Not Eligible, 18.39% Eligible, "
            "and 4.32% High Risk. Macro F1 is therefore more informative than "
            "accuracy alone.\n"
            "- Profiles without an existing loan were Eligible in 25.78% of rows, "
            "compared with 7.27% for profiles with an existing loan. This is an "
            "association in the supplied data, not proof of causation.\n"
            "- Vehicle and personal-loan scenarios had the highest Not Eligible "
            "shares, at 86.15% and 85.25%.\n"
            "- Median maximum monthly EMI was ₹13,840 for Eligible profiles, "
            "₹10,285 for High Risk profiles, and ₹2,464 for Not Eligible profiles."
        )

    with st.container(border=True):
        st.subheader("Potential business impact")
        st.markdown(
            "- Give staff a consistent first-pass affordability estimate.\n"
            "- Send High Risk cases to manual review instead of automatic rejection.\n"
            "- Use the maximum-EMI estimate to discuss safer requested amounts or tenures.\n"
            "- Compare model results with policy rules while keeping a human decision-maker.\n"
            "- Monitor performance and class balance as customer behaviour changes."
        )

    with st.container(border=True):
        st.subheader("Limitations and responsible use")
        st.markdown(
            "- Results depend on the supplied synthetic/project dataset and may not "
            "represent real applicants.\n"
            "- The saved model was trained in fast mode for understandable execution time.\n"
            "- Sensitive-group fairness, calibration, drift, security, and regulatory "
            "validation require additional work before any real lending use.\n"
            "- Predictions are educational decision support, not loan approval or "
            "financial advice."
        )


st.set_page_config(
    page_title="EMIPredict AI",
    page_icon=":material/account_balance:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(render_home, title="Home", icon=":material/home:", default=True),
        st.Page(
            render_assessment,
            title="Customer assessment",
            icon=":material/person_search:",
        ),
        st.Page(
            render_model_information,
            title="Model information",
            icon=":material/insights:",
        ),
        st.Page(
            render_project_report,
            title="Project report",
            icon=":material/description:",
        ),
    ],
    position="top",
)
page.run()
