
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
user_id = "puneet83" 
model_path = hf_hub_download(repo_id=f"{user_id}/tourism-churn-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Customer Tourism plan purchase Prediction App")
st.write("The Tourism plan purchase Prediction App is an internal tool for staff that predicts whether customers will accept the product or not.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")

# Collect user input
CityTier = st.selectbox("CityTier (City Tier of customer resides)", [1, 2, 3])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
DurationOfPitch = st.number_input("DurationOfPitch (Time spent on explainging the product with Customer )", value=10)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting (no of Person accompanied with Customer)", min_value=0.0, value=10000.0)
NumberOfFollowups = st.number_input("NumberOfFollowups (number of followups with Customer)", min_value=1, value=10)
PreferredPropertyStar = st.selectbox("PreferredPropertyStar", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("NumberOfTrips",  min_value=1, max_value=35, value=15)
Passport = st.selectbox("Passport (Customer Have Passport (0 - No, 1-Yes))", ["No","Yes"])
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("OwnCar (Customer Have OwnCar (0 - No, 1-Yes))", [0,1])
NumberOfChildrenVisiting = st.selectbox("NumberOfChildrenVisiting", [1, 2, 3, 4, 5])
MonthlyIncome = st.number_input("MonthlyIncome (customer’s estimated MonthlyIncome)", min_value=0.0, value=100000.0)

TypeofContact = st.selectbox("TypeofContact (How Customer Contacted)", ["Self Enquiry","Company Invited"])
Occupation = st.selectbox("Occupation (How Customer Contacted)", ["Salaried", "Free Lancer", "Small Business","Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("MaritalStatus (is Customer Married?)", ["Married","Unmarried","Single","Divorced"])

ProductPitched = st.selectbox("ProductPitched ", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
Designation = st.selectbox("Designation (Customer Designation)", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'CityTier': CityTier,
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'ProductPitched': ProductPitched,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
