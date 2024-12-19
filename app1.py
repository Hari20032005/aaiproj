import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from datetime import datetime

# Patient database
PATIENT_DATABASE = {
    "001": {
        "age": 40,
        "resting_bp": 140,
        "cholesterol": 289,
        "max_heart_rate": 172,
        "oldpeak": 0.0,
        "sex": "Male",
        "fasting_blood_sugar": "No",
        "exercise_angina": "No",
        "chest_pain_type": "Atypical Angina",  # Type 2
        "resting_ecg": "Normal",  # 0
        "st_slope": "Upsloping"  # 1
    },
    "002": {
        "age": 49,
        "resting_bp": 160,
        "cholesterol": 180,
        "max_heart_rate": 156,
        "oldpeak": 1.0,
        "sex": "Female",
        "fasting_blood_sugar": "No",
        "exercise_angina": "No",
        "chest_pain_type": "Non-anginal Pain",  # Type 3
        "resting_ecg": "Normal",  # 0
        "st_slope": "Flat"  # 2
    }
}

# [Previous preprocessing columns and create_features function remain the same]
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

    # Initialize session state for form values if not exists
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {}

    # If patient ID exists in database and form values haven't been set
    if patient_id in PATIENT_DATABASE and not st.session_state.form_values:
        st.session_state.form_values = PATIENT_DATABASE[patient_id]
        
    st.write("Enter your medical information to check your heart disease risk.")

    # Load the trained model and preprocessing columns
    try:
        model = joblib.load('svm_heart_disease_model.joblib')
        scaler = joblib.load('scaler.joblib')
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
        age = st.number_input("Age", 
                            min_value=1, max_value=100, 
                            value=st.session_state.form_values.get('age', 45))
        sex = st.selectbox("Sex", 
                          options=["Male", "Female"],
                          index=["Male", "Female"].index(st.session_state.form_values.get('sex', "Male")))
        chest_pain_type = st.selectbox("Chest Pain Type", 
                                     options=["Typical Angina", "Atypical Angina",
                                             "Non-anginal Pain", "Asymptomatic"],
                                     index=["Typical Angina", "Atypical Angina",
                                           "Non-anginal Pain", "Asymptomatic"].index(
                                               st.session_state.form_values.get('chest_pain_type', "Typical Angina")))
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 
                                   min_value=80, max_value=200, 
                                   value=st.session_state.form_values.get('resting_bp', 120))
        cholesterol = st.number_input("Cholesterol (mg/dl)", 
                                    min_value=100, max_value=600, 
                                    value=st.session_state.form_values.get('cholesterol', 200))

    with col2:
        st.subheader("Medical Information")
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                         options=["Yes", "No"],
                                         index=["Yes", "No"].index(
                                             st.session_state.form_values.get('fasting_blood_sugar', "No")))
        resting_ecg = st.selectbox("Resting ECG Results",
                                 options=["Normal", "ST-T Wave Abnormality",
                                         "Left Ventricular Hypertrophy"],
                                 index=["Normal", "ST-T Wave Abnormality",
                                       "Left Ventricular Hypertrophy"].index(
                                           st.session_state.form_values.get('resting_ecg', "Normal")))
        max_heart_rate = st.number_input("Maximum Heart Rate",
                                       min_value=60, max_value=220,
                                       value=st.session_state.form_values.get('max_heart_rate', 150))
        exercise_angina = st.selectbox("Exercise Induced Angina",
                                     options=["Yes", "No"],
                                     index=["Yes", "No"].index(
                                         st.session_state.form_values.get('exercise_angina', "No")))
        oldpeak = st.number_input("ST Depression (Oldpeak)",
                                min_value=0.0, max_value=10.0,
                                value=st.session_state.form_values.get('oldpeak', 0.0))
        st_slope = st.selectbox("ST Slope",
                              options=["Upsloping", "Flat", "Downsloping"],
                              index=["Upsloping", "Flat", "Downsloping"].index(
                                  st.session_state.form_values.get('st_slope', "Upsloping")))

    # Reset form values when changing patients
    if st.button("Clear Form"):
        st.session_state.form_values = {}
        st.experimental_rerun()

    # [Rest of the code remains the same - categorical mappings and prediction logic]
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
                st.error("⚠️ Heart Disease")
            else:
                st.success("✅ No Heart Disease")

        with result_col2:
            st.metric(
                label="Risk Probability",
                value=f"{prediction_proba[0][1]:.1%}"
            )

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

        if len(st.session_state.predictions) > 1:
            st.subheader("Previous Predictions")
            history_df = pd.DataFrame(st.session_state.predictions[:-1])
            st.dataframe(history_df)

if __name__ == "__main__":
    main()