import streamlit as st
import joblib
import pandas as pd

# Load model and data from pickle file
try:
    model_data = joblib.load("hr_attrition_logistic.pkl")
    model = model_data.get('model')
    accuracy = model_data.get('accuracy')
    X_test = model_data.get('X_test')
    y_test = model_data.get('y_test')
    df = model_data.get('data')
    if not all([model, accuracy, X_test is not None, y_test is not None, df is not None]):
        st.error("Missing required keys in hr_attrition_logistic.pkl. Ensure it contains 'model', 'accuracy', 'X_test', 'y_test', and 'data'.")
        st.stop()
except FileNotFoundError:
    st.error("hr_attrition_logistic.pkl not found. Please ensure the model file is in the correct directory.")
    st.stop()

# Title
st.title("Employee Attrition Prediction")

# Input fields
st.header("Enter Employee Details")
age = st.number_input("Age", min_value=18, max_value=65, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
overtime = st.checkbox("Works Overtime")
department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
job_role = st.selectbox("Job Role", [
    "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director",
    "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
])
education_field = st.selectbox("Education Field", [
    "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"
])

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'JobSatisfaction': [job_satisfaction],
        'YearsAtCompany': [years_at_company],
        'Department_Research & Development': [1 if department == "Research & Development" else 0],
        'Department_Sales': [1 if department == "Sales" else 0],
        'EducationField_Life Sciences': [1 if education_field == "Life Sciences" else 0],
        'EducationField_Marketing': [1 if education_field == "Marketing" else 0],
        'EducationField_Medical': [1 if education_field == "Medical" else 0],
        'EducationField_Other': [1 if education_field == "Other" else 0],
        'EducationField_Technical Degree': [1 if education_field == "Technical Degree" else 0],
        'JobRole_Human Resources': [1 if job_role == "Human Resources" else 0],
        'JobRole_Laboratory Technician': [1 if job_role == "Laboratory Technician" else 0],
        'JobRole_Manager': [1 if job_role == "Manager" else 0],
        'JobRole_Manufacturing Director': [1 if job_role == "Manufacturing Director" else 0],
        'JobRole_Research Director': [1 if job_role == "Research Director" else 0],
        'JobRole_Research Scientist': [1 if job_role == "Research Scientist" else 0],
        'JobRole_Sales Executive': [1 if job_role == "Sales Executive" else 0],
        'JobRole_Sales Representative': [1 if job_role == "Sales Representative" else 0],
        'MaritalStatus_Married': [0],
        'MaritalStatus_Single': [0],
        'Gender_Male': [0],
        'OverTime_Yes': [1 if overtime else 0]
    })
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.write(f"Employee likely to leave (Probability: {probability * 100:.2f}%)")
    else:
        st.write(f"Employee likely to stay (Probability: {(1 - probability) * 100:.2f}%)")

# Model evaluation
st.header("Model Performance")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

