import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write(
    """
   This app predicts whether a telecom customer is likely to churn based on their account and usage details. Enter the customer information and click Predict Churn.
    """
)

# ---------------------------------------------------------------
# Load the trained pipeline
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Could not find `churn_pipeline.pkl`. Make sure you have run "
        "`model_training.ipynb` first and that the file is in the same "
        "folder as `app.py`."
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------
# Collect user inputs for every model feature
# ---------------------------------------------------------------
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    tenure_months = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    contract_type = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])

with col2:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0
    )
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=20000.0, value=840.0, step=10.0
    )
    num_support_calls = st.slider("Number of Support Calls", 0, 15, 1)
    late_payments_last_year = st.slider("Late Payments (last year)", 0, 12, 0)
    avg_monthly_usage_gb = st.number_input(
        "Avg. Monthly Usage (GB)", min_value=0.0, max_value=2000.0, value=150.0, step=5.0
    )

st.divider()

# ---------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------
if st.button("Predict Churn", type="primary", use_container_width=True):
    try:
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "region": region,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "num_support_calls": num_support_calls,
            "late_payments_last_year": late_payments_last_year,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
        }])

        prediction = model.predict(input_df)[0]
        result_label = "Yes" if prediction == 1 else "No"

        if result_label == "Yes":
            st.error(f"### Prediction: **{result_label}** — this customer is likely to churn.")
        else:
            st.success(f"### Prediction: **{result_label}** — this customer is likely to stay.")

        # Bonus: show prediction probability/confidence if supported
        if hasattr(model.named_steps["classifier"], "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            churn_prob = proba[1]
            st.write("**Prediction confidence:**")
            st.progress(float(churn_prob))
            st.write(f"Probability of churn: **{churn_prob:.1%}**  |  Probability of staying: **{1 - churn_prob:.1%}**")

    except Exception as e:
        st.error(f"Something went wrong while making the prediction: {e}")

st.divider()
st.caption(
    "Model trained in `model_training.ipynb` using a Scikit-Learn "
    "ColumnTransformer + classifier Pipeline, saved as `churn_pipeline.pkl`."
)
