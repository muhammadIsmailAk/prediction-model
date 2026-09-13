# Customer Churn Prediction — End-to-End ML Project

Predicts whether a telecom customer will **churn** (`Yes`/`No`) using account,
billing, and usage data, and serves the trained model through a Streamlit web
app.

## 🎯 Project Overview

| Step | What was done |
|---|---|
| Data understanding | Loaded `customer_churn_data.csv`, inspected shape, dtypes, and summary statistics |
| Data quality checks | Found missing values, duplicate rows, unique category values, and class imbalance |
| Data cleaning | Dropped duplicates; missing values imputed inside the pipeline (median for numeric, most-frequent for categorical); `customer_id` excluded from features |
| EDA | Churn distribution, churn by contract type, churn by internet service, charges/tenure distributions, support-call analysis, correlation heatmap |
| Preprocessing | `ColumnTransformer` with `SimpleImputer` + `StandardScaler` (numeric) and `SimpleImputer` + `OneHotEncoder` (categorical) |
| Modeling | Trained **Logistic Regression** and **Random Forest**, compared on Accuracy / Precision / Recall / F1 |
| Model selection | Best model chosen by **F1 Score** on the held-out test set (not training accuracy) |
| Deployment | Full pipeline saved as `churn_pipeline.pkl`, reloaded and validated, then served via a Streamlit app |

## 📁 Repository Structure

```
├── customer_churn_data.csv   # Dataset
├── model_training.ipynb      # Full training notebook (Part A)
├── churn_pipeline.pkl        # Saved preprocessing + model pipeline
├── app.py                    # Streamlit web app (Part B)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 📊 Dataset

17 columns including customer demographics (`age`, `gender`, `region`),
account details (`tenure_months`, `contract_type`, `payment_method`),
service usage (`internet_service`, `avg_monthly_usage_gb`), support history
(`num_support_calls`, `late_payments_last_year`), and the target `churn`.

> **Note:** This repository's `customer_churn_data.csv` is a synthetic
> dataset generated to match the required schema for demonstration. If you
> have the original dataset, replace the CSV with it and re-run
> `model_training.ipynb` — no code changes are required since the pipeline is
> schema-driven, not hard-coded to specific values.

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Re)train the model** (optional — `churn_pipeline.pkl` is already included)
   ```bash
   jupyter notebook model_training.ipynb
   ```
   Run all cells top to bottom. This regenerates `churn_pipeline.pkl`.

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   Open the local URL Streamlit prints (usually `http://localhost:8501`).

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (including `churn_pipeline.pkl`,
   `app.py`, and `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select this repository and branch, set the main file
   path to `app.py`, and click **Deploy**.
4. Once deployed, test the public app URL with a few sample inputs.

## 🧠 Model Details

- **Algorithm(s) compared:** Logistic Regression, Random Forest
- **Selection metric:** F1 Score (balances precision/recall under class
  imbalance — more relevant than raw accuracy for churn use cases)
- **Preprocessing:** Median imputation + scaling (numeric), most-frequent
  imputation + one-hot encoding (categorical), all inside one
  `sklearn.Pipeline` so training-time and prediction-time preprocessing are
  guaranteed identical.

## ⚠️ Limitations & Next Steps

- Trained on a moderate-sized dataset; performance may shift with more data.
- Class imbalance (more non-churners than churners) means recall on the
  minority ("Yes") class deserves ongoing monitoring in production.
- Potential improvements: hyperparameter tuning (`GridSearchCV`), trying
  gradient boosting models, SHAP-based feature importance for
  interpretability, and periodic retraining as customer behavior evolves.

## 📄 License

Educational project — for coursework/demonstration purposes.
