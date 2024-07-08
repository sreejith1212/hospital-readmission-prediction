import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

    
def readmission_predictor(age, Num_Lab_Procedures, Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits, 
                            Num_Diagnoses, Num_Emergency_Visits, gender, admission_type, diagnosis, a1c_result):

    # Create DataFrame using the provided data
    unseen_data = pd.DataFrame({
        'Age': [age],
        'Num_Lab_Procedures': [Num_Lab_Procedures],
        'Num_Medications': [Num_Medications],
        'Num_Outpatient_Visits': [Num_Outpatient_Visits],
        'Num_Inpatient_Visits': [Num_Inpatient_Visits],
        'Num_Diagnoses': [Num_Diagnoses],
        'Num_Emergency_Visits': [Num_Emergency_Visits],
        'Gender': [gender],
        'Admission_Type': [admission_type],
        'Diagnosis': [diagnosis],
        'A1C_Result': [a1c_result]
    })

    with open(r"E:\GUVI Main Boot\final capstone project\gbr_model.pkl", "rb") as file:
        prediction_model = pickle.load(file)
    with open(r"E:\GUVI Main Boot\final capstone project\encoded_columns.pkl", "rb") as file:
        encoder_data = pickle.load(file)
    with open(r"E:\GUVI Main Boot\final capstone project\scalers.pkl", "rb") as file:
        scaler_data = pickle.load(file)

    # One-hot encode unseen data
    unseen_encoded = pd.get_dummies(unseen_data, dtype=int)

    # Reindex unseen data to match the training data columns
    unseen_encoded = unseen_encoded.reindex(columns=encoder_data, fill_value=0)

    to_scale_features = ['Age', 'Num_Lab_Procedures', 'Num_Medications', 'Num_Outpatient_Visits', 'Num_Inpatient_Visits', 'Num_Emergency_Visits', 'Num_Diagnoses']
    # Transform the unseen data
    for col in to_scale_features:
        unseen_encoded[col] = scaler_data[col].transform(unseen_encoded[[col]])

    req_col = ['Num_Diagnoses_Group_Low', 'Num_Medications_Group_High', 'Num_Diagnoses_Group_Medium', 'Diagnosis_Heart Disease', 'Num_Inpatient_Visits', 'A1C_Result_Normal']


    data = unseen_encoded[req_col].values

    input_predict_readmission = prediction_model.predict(data)

    return input_predict_readmission

if __name__ == "__main__":

    # set app page layout type
    st.set_page_config(layout="wide")

    # create sidebar
    with st.sidebar:        
        page = option_menu(
                            menu_title='Hospital App',
                            options=['Home', 'Predict Readmission'],
                            icons=['gear', 'bar-chart-line'], 
                            menu_icon="pin-map-fill",
                            default_index=0 ,
                            styles={"container": {"padding": "5!important"},
                                    "icon": {"color": "brown", "font-size": "23px"}, 
                                    "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "lightblue", "display": "flex", 
                                                 "align-items": "center"},
                                    "nav-link-selected": {"background-color": "grey"},}  
        )


if page == "Home":

    st.header("Hospital Readmission Prediction", divider = "rainbow")
    st.write("")

    st.subheader(":orange[Application Property :]")
    st.subheader(":one: :grey[_Identify patients who are at high risk of hospital readmission within 30 days after their initial discharge_.]")
            
if page == "Predict Readmission":

    col1, col2, col3 = st.columns([1,2,1])
    col2.header(':green[Hospital Readmisssion Prediction] 🏥')
    container_upload = st.container(height=600, border=False)
    for i in range(3):
        container_upload.write("")
    with container_upload.form(key = "readmit"):
        col14, col15, col16 = st.columns(3)

        age = col14.number_input(label="Age 🌃", value=0)
        Num_Lab_Procedures = col15.selectbox(label="Number Of Lab Procedures 🛹", options=tuple(range(0,100)))
        Num_Medications = col16.selectbox(label="Number Of Medications 🏟️", options=tuple(range(0,36)))
        Num_Outpatient_Visits = col14.selectbox(label="Number Of Outpatient Visits 📱", options=tuple(range(0,50)))
        Num_Inpatient_Visits = col15.selectbox(label="Number Of Inpatient Visits 🏫", options=tuple(range(0,50)))
        Num_Diagnoses = col16.selectbox(label="Number Of Diagnosis 🏯", options=tuple(range(0,10)))
        Num_Emergency_Visits = col14.selectbox(label="Number Of Emergency Visits 🌆", options=tuple(range(0,50)))
        gender = col15.selectbox(label="Gender ⚙️", options=['Male', 'Female', 'Other'])
        admission_type = col16.selectbox(label="Admission Type 🌐", options=['Emergency', 'Urgent', 'Elective'])
        diagnosis = col14.selectbox(label="Diagnosis 💺", options=['Heart Disease', 'Diabetes', 'Injury', 'Infection'])
        a1c_result = col15.selectbox(label="A1C Result 📧", options=['Normal', 'Abnormal', 'Unknown'])

        upload_data = col14.form_submit_button(label="Predict Readmission", help="Click to Predict Hospital Readmission", type = "primary")
    
    col4, col5, col6 = st.columns(3)
    if upload_data:
        try:
            prediction_1 = readmission_predictor(age, Num_Lab_Procedures, Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits, 
                                                 Num_Diagnoses, Num_Emergency_Visits, gender, admission_type, diagnosis, a1c_result)
            
            if prediction_1 == 1:
                col4.error(f"Readmission Requirement : Required ❗")
            else:
                col4.success(f"Readmission : Not Required 🀄")
        except:
            col4.error("Enter valid values 🚨")

    

            