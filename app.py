import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --- Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Predictor (XGBoost)",
    layout="centered"
)

# Columns that contain 0 values that need to be replaced with the median
IMPUTE_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
TARGET_COL = 'Outcome'
RANDOM_SEED = 42

# --- Data Loading and Model Training (Cached) ---

@st.cache_data
def load_data():
    """
    Loads the Pima Indians Diabetes dataset. In a real deployment, 
    you would load this from a local file, cloud storage, or database.
    We use a public link for a self-contained, runnable example.
    """
    # CORRECTED URL: Using a more stable GitHub mirror for the dataset.
    DATA_URL ="C:\Users\Arun kumar\Downloads\boosting diabetes xgb\diabetes.csv"
    try:
        # Note: Removed skiprows=1 as the new source file does not have a header.
        data = pd.read_csv(DATA_URL,
                           names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    except Exception as e:
        st.error(f"Could not load data. Please ensure 'pima-indians-diabetes.csv' is available. Error: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure
    return data

@st.cache_resource
def train_model(data):
    """
    Trains the XGBoost model and calculates median values for imputation, 
    matching the logic in your notebook.
    """
    if data.empty:
        return None, None

    # Create a copy to perform preprocessing
    df_processed = data.copy()

    # Calculate medians BEFORE imputation (to use on user input)
    medians = df_processed[IMPUTE_COLS].replace(0, np.nan).median()

    # Apply preprocessing (replace 0s with median)
    for col in IMPUTE_COLS:
        # Use the calculated median to fill zeros
        df_processed[col] = df_processed[col].replace(0, medians[col])

    # Separate features (X) and target (y)
    X = df_processed.drop(TARGET_COL, axis=1)
    y = df_processed[TARGET_COL]

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Initialize and train the XGBoost Classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    return model, medians

# Load data and train model
data = load_data()
model, medians = train_model(data)

# --- Prediction Function ---

def predict_risk(input_data, model, medians):
    """
    Applies the same preprocessing (median imputation) to the user input 
    and returns the prediction and probability.
    """
    if model is None or medians is None:
        return "Model not initialized.", 0.0

    # Convert input to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply median imputation to the input data for the specified columns
    for col in IMPUTE_COLS:
        if input_df[col].iloc[0] == 0:
            input_df[col] = medians[col]

    # Make prediction (0 or 1) and prediction probability
    prediction = model.predict(input_df)[0]
    # Get the probability for the positive class (Diabetes: 1)
    probability = model.predict_proba(input_df)[0][1] 

    return prediction, probability

# --- Streamlit UI Layout ---

st.title("Pima Indians Diabetes Risk Predictor")
st.markdown("""
This application uses an **XGBoost Classifier** model, mirroring the development in your Jupyter notebook, to assess the risk of diabetes based on key health indicators.
""")

if data.empty or model is None:
    st.error("Application setup failed. Please check the data loading or model training steps.")
else:
    # --- Sidebar for Inputs ---
    st.sidebar.header("Input Patient Data")

    # Define the input fields and their sensible ranges
    pregnancies = st.sidebar.slider("1. Pregnancies (Number of times pregnant)", 0, 17, 3)
    glucose = st.sidebar.slider("2. Glucose (Plasma glucose concentration, mg/dl)", 0, 200, 117)
    blood_pressure = st.sidebar.slider("3. Blood Pressure (Diastolic, mmHg)", 0, 122, 72)
    skin_thickness = st.sidebar.slider("4. Skin Thickness (Triceps skin fold thickness, mm)", 0, 99, 29)
    insulin = st.sidebar.slider("5. Insulin (2-Hour serum insulin, mu U/ml)", 0, 846, 79)
    bmi = st.sidebar.slider("6. BMI (Body Mass Index)", 0.0, 67.1, 32.0)
    dpf = st.sidebar.number_input("7. Diabetes Pedigree Function", 0.078, 2.42, 0.3725, step=0.01, format="%.4f")
    age = st.sidebar.slider("8. Age (Years)", 21, 81, 29)


    # Assemble input data dictionary in the order of the features used in training
    user_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    # --- Main Content: Prediction ---
    st.header("Prediction Results")

    # Button to trigger prediction
    if st.sidebar.button("Predict Risk", use_container_width=True, type="primary"):
        with st.spinner('Analyzing data and running prediction...'):
            prediction, probability = predict_risk(user_input, model, medians)

            risk_percentage = probability * 100

            if prediction == 1:
                st.error(f"High Risk of Diabetes (Class 1)")
                st.markdown(f"**Probability of having Diabetes:** **{risk_percentage:.2f}%**")
                st.warning("The model predicts a high likelihood of a positive outcome (Diabetes). It is highly recommended to consult a healthcare professional for a confirmed diagnosis.")
                
            else:
                st.success(f"Low Risk of Diabetes (Class 0)")
                st.markdown(f"**Probability of having Diabetes:** **{risk_percentage:.2f}%**")
                st.info("The model predicts a low likelihood of a positive outcome (Diabetes). Remember that this is only a statistical prediction and not a medical diagnosis.")

            # Display the processed input values for transparency
            st.subheader("Input Details for Prediction")
            
            # Show how the input was processed (especially 0s)
            processed_input_df = pd.DataFrame([user_input])
            for col in IMPUTE_COLS:
                if processed_input_df[col].iloc[0] == 0:
                    processed_input_df[col] = f"{medians[col]:.2f} (Imputed Median)"
                else:
                    processed_input_df[col] = f"{processed_input_df[col].iloc[0]:.2f}"


            st.dataframe(processed_input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    st.markdown("---")
    st.markdown("Enter the patient's data in the sidebar and click 'Predict Risk'.")

# Footer
st.markdown("---")
st.caption("Model trained using XGBoost on a standard diabetes dataset.")
