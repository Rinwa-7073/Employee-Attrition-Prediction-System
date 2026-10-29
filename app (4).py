import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction Dashboard")
st.caption("AIDAS 2 Mini Project | Machine Learning")
st.write(
    "Enter the employee details to get a model-based estimate of attrition. "
    "The app uses only the features present in the project's cleaned dataset."
)

MODEL_PATH = "employee_attrition_model.pkl"
DATA_PATH = "employee_attrition_dataset.csv"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run all cells in `Employee_Attrition_Prediction.ipynb` "
        "first. It will create `employee_attrition_model.pkl` in this folder."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
medians = bundle["medians"]
modes = bundle["modes"]
model_name = bundle.get("selected_model", "Trained model")

reference_df = None
if os.path.exists(DATA_PATH):
    try:
        reference_df = pd.read_csv(DATA_PATH)
    except Exception:
        reference_df = None

with st.sidebar:
    st.header("Project Details")
    st.write("**Project:** Employee Attrition Prediction System")
    st.write("**Model:**", model_name)
    st.write("**Target:** Attrition (Yes / No)")
    st.write("**Dataset records:**", len(reference_df) if reference_df is not None else "Not loaded")
    st.info(
        "Educational demonstration only. Predictions are estimates and should not "
        "be used as the sole basis for employment decisions."
    )

st.subheader("Employee Information")
st.write("Complete the fields below, then select **Predict Attrition**.")

input_values = {}
with st.form("attrition_form"):
    left, right = st.columns(2)
    for idx, feature in enumerate(feature_columns):
        panel = left if idx % 2 == 0 else right
        with panel:
            if feature in numeric_features:
                default = medians.get(feature, 0)
                try:
                    default = float(default)
                except (TypeError, ValueError):
                    default = 0.0

                # These are discrete integer-valued fields in the supplied dataset.
                if feature in ["Age", "DailyRate", "DistanceFromHome", "Education"]:
                    min_value = 0 if feature != "Age" else 18
                    input_values[feature] = st.number_input(
                        feature,
                        min_value=min_value,
                        value=max(min_value, int(round(default))),
                        step=1
                    )
                else:
                    input_values[feature] = st.number_input(
                        feature, value=float(default), step=1.0
                    )
            else:
                if reference_df is not None and feature in reference_df.columns:
                    options = sorted(reference_df[feature].dropna().astype(str).unique().tolist())
                else:
                    options = []

                default_value = str(modes.get(feature, options[0] if options else "Unknown"))
                if default_value not in options and options:
                    default_value = options[0]
                if not options:
                    options = [default_value or "Unknown"]
                default_index = options.index(default_value) if default_value in options else 0

                input_values[feature] = st.selectbox(
                    feature, options=options, index=default_index
                )

    submitted = st.form_submit_button(
        "Predict Attrition", type="primary", use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    for col in numeric_features:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
    for col in categorical_features:
        input_df[col] = input_df[col].astype(str)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.divider()
    st.subheader("Prediction Result")
    c1, c2 = st.columns(2)
    c1.metric("Estimated attrition probability", f"{probability:.1%}")
    c2.metric("Model prediction", "Attrition: Yes" if prediction == 1 else "Attrition: No")

    st.progress(min(max(probability, 0.0), 1.0))
    if prediction == 1:
        st.warning(
            "The model predicts the Attrition = Yes class for this profile. "
            "This is a statistical estimate, not a statement about an actual employee's intentions."
        )
    else:
        st.success(
            "The model predicts the Attrition = No class for this profile. "
            "This is an estimate and not a guarantee."
        )

    with st.expander("View submitted details"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Employee Attrition Prediction System | AIDAS 2 Mini Project")
