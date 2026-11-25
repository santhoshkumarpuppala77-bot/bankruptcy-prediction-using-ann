
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Corporate Bankruptcy Prediction using ANN

st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Corporate Bankruptcy Prediction")
st.subheader("Using Artificial Neural Networks")

st.write(
    "Enter the financial details of a company "
    "to predict its bankruptcy status."
)

MODEL_PATH = "model.keras"
PREPROCESSOR_PATH = "preprocessor"


# Load Model and Preprocessor

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


try:
    model, preprocessor = load_artifacts()

except Exception as error:
    st.error("Could not load model or preprocessor.")
    st.exception(error)
    st.stop()

years = list(range(1999, 2019))
divisions = ['D', 'I', 'E', 'G', 'B', 'F', 'H', 'C', 'A', 'J']

# Input Form

st.divider()
st.markdown("### Company Financial Details")

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
        "Predict Bankruptcy",
        use_container_width=True
    )


# Prediction

if submitted:

    input_data = pd.DataFrame([{

    "fyear":int(np.random.choice(years)),
    "Division": np.random.choice(divisions),

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
    "MajorGroup": major_group

}])

    try:

        processed_data = preprocessor.transform(input_data)

        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

        processed_data = np.asarray(
            processed_data,
            dtype=np.float32
        )

        prediction = model.predict(
            processed_data,
            verbose=0
        )

        probability = float(prediction.reshape(-1)[0])

        predicted_class = 1 if probability >= 0.5 else 0

        st.divider()
        st.subheader("Prediction Result")

        if predicted_class == 0:

            st.success("ALIVE")

            st.write(
                "The model predicts that the company is Alive."
            )

        else:

            st.error("FAILED")

            st.write(
                "The model predicts that the company is Failed."
            )

        st.write(f"Predicted Class: **{predicted_class}**")

    except Exception as error:

        st.error("Prediction could not be completed.")
        st.exception(error)
