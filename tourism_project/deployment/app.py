import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.utils.multiclass import type_of_target

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_package_prediction.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package based on customer interaction and demographic data.
""")

age = st.number_input("Age", 18, 100, 35)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (mins)", 0, 150, 15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
no_of_person_visiting = st.number_input("Number of Person Visiting", 1, 10, 3)
no_of_followups = st.number_input("Number of Followups", 1, 10, 3)
product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
pref_prop_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
no_of_trips = st.number_input("No of Trips", 1, 30, 3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
passport = st.selectbox("Passport (0: No, 1: Yes)", [0, 1])
pitch_satisf_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own a Car (0: No, 1: Yes)", [0, 1])
no_of_children_visiting = st.number_input("Number of Children Visiting", 0, 10, 1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 1000, 200000, 22000)

input_data = pd.DataFrame([{
    "Age": age,
    "Type of Contact": type_of_contact,
    "City Tier": city_tier,
    "Duration Of Pitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "Number Of Person Visiting": no_of_person_visiting,
    "Number Of Followups": no_of_followups,
    "Product Pitched": product_pitched,
    "Preferred Property Star": pref_prop_star,
    "Number Of Trips": no_of_trips,
    "Marital Status": marital_status,
    "Passport": passport,
    "Pitch Satisfaction Score": pitch_satisf_score,
    "Own Car": own_car,
    "Number Of Children Visiting": no_of_children_visiting,
    "Designation": designation,
    "Monthly Income": monthly_income
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: **{result}**")
    else:
        st.warning(f"The model predicts: **{result}**")
