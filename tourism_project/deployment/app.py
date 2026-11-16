"""
Streamlit app: Tourism Package Purchase Prediction
--------------------------------------------------
Loads the model committed by the GitHub Actions pipeline (it sits next to this
file), collects a customer's profile and interaction details, assembles them
into a single-row DataFrame with the exact feature names the training pipeline
used, and predicts whether the customer is likely to purchase the Wellness
Tourism Package. Streamlit Community Cloud runs this file directly from the repo.
"""
import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(
    os.path.dirname(__file__), "best_tourism_package_model_v1.joblib"
)
model = joblib.load(model_path)

st.set_page_config(page_title="Tourism Package Prediction", page_icon="🧳")
st.title("Tourism Package Purchase Prediction")
st.write(
    "Fill in the customer details below to predict whether they are likely to "
    "purchase the Wellness Tourism Package."
)

# ---- Customer profile ----
st.subheader("Customer profile")
Age = st.slider("Age", 18, 70, 35)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox(
    "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
)
Designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
MonthlyIncome = st.number_input(
    "Monthly Income", min_value=1000.0, max_value=100000.0, value=20000.0, step=500.0
)
Passport = st.selectbox("Holds a Passport", [0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No")
OwnCar = st.selectbox("Owns a Car", [0, 1],
                      format_func=lambda x: "Yes" if x == 1 else "No")

# ---- Trip details ----
st.subheader("Trip details")
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 6, 3)
NumberOfChildrenVisiting = st.slider("Number of Children Visiting (under 5)", 0, 3, 0)
NumberOfTrips = st.slider("Average Number of Trips per Year", 1, 22, 3)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])

# ---- Interaction details ----
st.subheader("Sales interaction")
ProductPitched = st.selectbox(
    "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)
DurationOfPitch = st.slider("Duration of Pitch (minutes)", 0, 100, 15)
NumberOfFollowups = st.slider("Number of Follow-ups", 0, 6, 3)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Assemble one row with the exact feature names used in training
input_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict"):
    pred = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0][1])
    if pred == 1:
        st.success(f"Likely to PURCHASE the package  (probability {proba:.1%})")
    else:
        st.info(f"Unlikely to purchase the package  (probability {proba:.1%})")
    st.caption(
        "Probability is the model's estimated likelihood that the customer "
        "purchases the Wellness Tourism Package."
    )
