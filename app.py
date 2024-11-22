import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from datetime import datetime

# Default preprocessing columns if the file is not available
DEFAULT_PREPROCESSING_COLUMNS = [
    'age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak',
    'sex', 'fasting blood sugar', 'exercise angina', 'chest pain type',
    'resting ecg', 'ST slope',
    'age_group_very_young', 'age_group_young', 'age_group_middle',
    'age_group_senior', 'age_group_elderly',
    'bp_category_normal', 'bp_category_prehypertension',
    'bp_category_stage1', 'bp_category_stage2'
]

def create_features(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], labels=['very_young', 'young', 'middle', 'senior', 'elderly'])
    df['bp_category'] = pd.cut(df['resting bp s'], bins=[0, 120, 140, 160, float('inf')], labels=['normal', 'prehypertension', 'stage1', 'stage2'])
    df = pd.get_dummies(df, columns=['age_group', 'bp_category'])
    return df

def preprocess_new_data(new_data, preprocessing_columns):
    new_data = create_features(new_data)
    for col in preprocessing_columns:
        if col not in new_data.columns:
            new_data[col] = 0
    return new_data[preprocessing_columns]

def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
    
    st.title("❤️ Heart Disease Risk Prediction")
    
    # Get patient ID input
    patient_id = st.text_input("Enter Patient ID:")
    
    if not patient_id:
        st.warning("Please enter a Patient ID to proceed.")
        return
        
    st.write("Enter your medical information to check your heart disease risk.")

    # Load the trained model and preprocessing columns
    try:
        model = joblib.load('svm_heart_disease_model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Try to load preprocessing columns, use default if file doesn't exist
        try:
            preprocessing_columns = joblib.load('preprocessing_columns.joblib')
        except FileNotFoundError:
            preprocessing_columns = DEFAULT_PREPROCESSING_COLUMNS
            st.info("Using default preprocessing columns.")
            
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model is trained and saved properly.")
        return

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        chest_pain_type = st.selectbox("Chest Pain Type", 
                                     options=["Typical Angina", 
                                             "Atypical Angina",
                                             "Non-anginal Pain",
                                             "Asymptomatic"],
                                     help="Type of chest pain experienced")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 
                                   min_value=80, max_value=200, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 
                                    min_value=100, max_value=600, value=200)

    with col2:
        st.subheader("Medical Information")
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                         options=["Yes", "No"])
        resting_ecg = st.selectbox("Resting ECG Results",
                                 options=["Normal",
                                         "ST-T Wave Abnormality",
                                         "Left Ventricular Hypertrophy"])
        max_heart_rate = st.number_input("Maximum Heart Rate",
                                       min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina",
                                     options=["Yes", "No"])
        oldpeak = st.number_input("ST Depression (Oldpeak)",
                                min_value=0.0, max_value=10.0, value=0.0)
        st_slope = st.selectbox("ST Slope",
                              options=["Upsloping", "Flat", "Downsloping"])

    # Create a dictionary to map categorical values to numerical ones
    categorical_mappings = {
        "sex": {"Male": 1, "Female": 0},
        "fasting_blood_sugar": {"Yes": 1, "No": 0},
        "exercise_angina": {"Yes": 1, "No": 0},
        "chest_pain_type": {
            "Typical Angina": 1,
            "Atypical Angina": 2,
            "Non-anginal Pain": 3,
            "Asymptomatic": 4
        },
        "resting_ecg": {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        },
        "st_slope": {
            "Upsloping": 1,
            "Flat": 2,
            "Downsloping": 3
        }
    }

    if st.button("Predict Heart Disease Risk", type="primary"):
        # Create DataFrame with user inputs
        user_data = pd.DataFrame({
            'age': [age],
            'resting bp s': [resting_bp],
            'cholesterol': [cholesterol],
            'max heart rate': [max_heart_rate],
            'oldpeak': [oldpeak],
            'sex': [categorical_mappings["sex"][sex]],
            'fasting blood sugar': [categorical_mappings["fasting_blood_sugar"][fasting_blood_sugar]],
            'exercise angina': [categorical_mappings["exercise_angina"][exercise_angina]],
            'chest pain type': [categorical_mappings["chest_pain_type"][chest_pain_type]],
            'resting ecg': [categorical_mappings["resting_ecg"][resting_ecg]],
            'ST slope': [categorical_mappings["st_slope"][st_slope]]
        })

        # Preprocess the data
        processed_data = preprocess_new_data(user_data, preprocessing_columns)
        scaled_data = scaler.transform(processed_data)

        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        # Display results
        st.markdown("---")
        st.subheader(f"Prediction Results for Patient ID: {patient_id}")

        # Create columns for the results
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction[0] == 1:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

        with result_col2:
            st.metric(
                label="Risk Probability",
                value=f"{prediction_proba[0][1]:.1%}"
            )

        # Store prediction in session state
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
            
        st.session_state.predictions.append({
            'patient_id': patient_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prediction': "High Risk" if prediction[0] == 1 else "Low Risk",
            'probability': f"{prediction_proba[0][1]:.1%}"
        })

        st.markdown("---")
        st.info("""
        **Note:** This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice. 
        Please consult with a healthcare provider for proper diagnosis and treatment.
        """)

        # Display prediction history
        if len(st.session_state.predictions) > 1:
            st.subheader("Previous Predictions")
            history_df = pd.DataFrame(st.session_state.predictions[:-1])
            st.dataframe(history_df)

if __name__ == "__main__":
    main()