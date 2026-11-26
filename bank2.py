from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="📊",
    layout="centered",
)

# ==========================================
# 2. CUSTOM BACKGROUND AND CSS
# ==========================================

st.markdown(
    """
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #172554,
        #1e1b4b
    );
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

h1 {
    color: #60a5fa !important;
    text-align: center;
    font-weight: 800 !important;
}

h2, h3 {
    color: #93c5fd !important;
}

p, label, .stMarkdown {
    color: #e2e8f0;
}

div[data-baseweb="input"] input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 8px;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(96,165,250,0.3);
}

.stFormSubmitButton button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: bold;
    transition: 0.3s ease;
}

.stFormSubmitButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 20px rgba(59,130,246,0.4);
}

hr {
    border-color: rgba(147,197,253,0.25);
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("🏢 Corporate Bankruptcy Prediction")

st.markdown(
    "<h3 style='text-align:center;'>"
    "Artificial Neural Network Based Prediction System"
    "</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>"
    "Analyze financial indicators and predict corporate financial status."
    "</p>",
    unsafe_allow_html=True,
)


# ==========================================
# 4. MODEL AND PREPROCESSOR
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor"


def _patch_sklearn_remainder_cols_list() -> None:
    """Compatibility shim for ColumnTransformer pickles created in older scikit-learn."""
    try:
        from sklearn.compose import _column_transformer as ct_module

        if not hasattr(ct_module, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct_module._RemainderColsList = _RemainderColsList
    except Exception:
        pass


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file at: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessor file at: {PREPROCESSOR_PATH}")

    _patch_sklearn_remainder_cols_list()

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    preprocessor = joblib.load(
        str(PREPROCESSOR_PATH)
    )

    return model, preprocessor


try:
    model, preprocessor = load_artifacts()
except Exception as error:
    st.error(
        "Could not load the model or preprocessor. "
        "Check that both 'model.keras' and 'preprocessor' are in the repository root folder."
    )
    st.exception(error)
    st.stop()


# ==========================================
# 5. RANDOM YEAR AND DIVISION
# ==========================================

years = list(range(1999, 2019))

divisions = [
    "D", "I", "E", "G", "B",
    "F", "H", "C", "A", "J",
]


# ==========================================
# 6. FINANCIAL INPUT FORM
# ==========================================

st.divider()

st.markdown("### 📋 Company Financial Details")

with st.form("bankruptcy_form"):
    col1, col2 = st.columns(2)

    with col1:
        current_assets = st.number_input("Current Assets", value=0.0)
        cost_of_goods_sold = st.number_input("Cost of Goods Sold", value=0.0)
        depreciation = st.number_input("Depreciation and Amortization", value=0.0)
        ebitda = st.number_input("EBITDA", value=0.0)
        inventory = st.number_input("Inventory", value=0.0)
        net_income = st.number_input("Net Income", value=0.0)
        total_receivables = st.number_input("Total Receivables", value=0.0)
        market_value = st.number_input("Market Value", value=0.0)
        net_sales = st.number_input("Net Sales", value=0.0)

    with col2:
        total_assets = st.number_input("Total Assets", value=0.0)
        long_term_debt = st.number_input("Long Term Debt", value=0.0)
        ebit = st.number_input("EBIT", value=0.0)
        gross_profit = st.number_input("Gross Profit", value=0.0)
        current_liabilities = st.number_input("Current Liabilities", value=0.0)
        retained_earnings = st.number_input("Retained Earnings", value=0.0)
        total_revenue = st.number_input("Total Revenue", value=0.0)
        total_liabilities = st.number_input("Total Liabilities", value=0.0)
        operating_expenses = st.number_input("Operating Expenses", value=0.0)

    major_group = st.number_input("Major Group", value=0.0)

    submitted = st.form_submit_button(
        "🔍 Predict Bankruptcy",
        use_container_width=True,
    )


# ==========================================
# 7. PREDICTION
# ==========================================

if submitted:
    # Randomly select year and division
    fyear = int(np.random.choice(years))
    division = str(np.random.choice(divisions))

    # Display randomly selected values
    st.info(
        f"Demonstration inputs: Financial Year = {fyear} | "
        f"Division = {division}"
    )

    # Create input DataFrame matching training feature names
    input_data = pd.DataFrame(
        [
            {
                "fyear": fyear,
                "Division": division,
                "Current_Assets": current_assets,
                "Cost_of_Goods_Sold": cost_of_goods_sold,
                "Depreciation_and_Amortization": depreciation,
                "EBITDA": ebitda,
                "Inventory": inventory,
                "Net_Income": net_income,
                "Total_Receivables": total_receivables,
                "Market_Value": market_value,
                "Net_Sales": net_sales,
                "Total_Assets": total_assets,
                "Long_Term_Debt": long_term_debt,
                "EBIT": ebit,
                "Gross_Profit": gross_profit,
                "Current_Liabilities": current_liabilities,
                "Retained_Earnings": retained_earnings,
                "Total_Revenue": total_revenue,
                "Total_Liabilities": total_liabilities,
                "Operating_Expenses": operating_expenses,
                "MajorGroup": major_group,
            }
        ]
    )

    try:
        # Apply preprocessing
        processed_data = preprocessor.transform(input_data)

        # Convert sparse matrix if required
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

        processed_data = np.asarray(
            processed_data,
            dtype=np.float32,
        )

        expected_features = model.input_shape[-1]
        if expected_features is not None and processed_data.shape[1] != expected_features:
            st.error(
                f"Feature mismatch! Expected {expected_features} features, "
                f"but received {processed_data.shape[1]}."
            )
            st.stop()

        # ANN prediction
        prediction = model.predict(
            processed_data,
            verbose=0,
        )

        probability = float(prediction.reshape(-1)[0])
        predicted_class = 1 if probability >= 0.5 else 0

        # ==================================
        # 8. DISPLAY PREDICTION RESULT
        # ==================================

        st.divider()
        st.subheader("📊 Prediction Result")

        if predicted_class == 0:
            st.success("🟢 COMPANY STATUS: ALIVE")
            st.markdown(
                """
            <div style="
                background:rgba(22,163,74,0.15);
                padding:20px;
                border-radius:12px;
                border:1px solid #22c55e;
                text-align:center;
            ">
                <h2 style="color:#4ade80;">ALIVE</h2>
                <p>The model predicts that the company
                belongs to the Alive class.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.error("🔴 COMPANY STATUS: FAILED")
            st.markdown(
                """
            <div style="
                background:rgba(220,38,38,0.15);
                padding:20px;
                border-radius:12px;
                border:1px solid #ef4444;
                text-align:center;
            ">
                <h2 style="color:#f87171;">FAILED</h2>
                <p>The model predicts that the company
                belongs to the Failed class.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Predicted Class",
                value=predicted_class,
            )
        with col_m2:
            st.metric(
                label="Failure Probability",
                value=f"{probability:.2%}",
            )

        st.caption(
            "This prediction is generated by an experimental "
            "machine learning model and is not a guarantee "
            "of the company's financial future."
        )

    except Exception as error:
        st.error(
            "Prediction could not be completed. "
            "Please check the input values and model files."
        )
        st.exception(error)


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>"
    "Corporate Bankruptcy Prediction | ANN Machine Learning Project"
    "</p>",
    unsafe_allow_html=True,
)
