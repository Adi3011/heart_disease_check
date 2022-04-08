import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import sklearn

DATASET_PATH = "data/heart_2020.csv"
LOG_MODEL_PATH = "model/logistic.pkl"

heart=pd.read_csv(DATASET_PATH)

def main():
    def user_input_features():
        race = st.sidebar.selectbox("Race", options=('White', 'Black','Asian', 'American Indian/Alaskan Native','Other','Hispanic'))
        sex = st.sidebar.selectbox("Sex", options=('Female','Male'))
        age_cat = st.sidebar.selectbox("Age category",
                                       options=('18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'))
        bmi_cat = st.sidebar.selectbox("BMI category",
                                       options=('Normal(BMI > 18.5 and <25.5)','Underweight (BMI <= 18.5)','Overweight(BMI >=25.5 and BMI<30.0)','Obese (BMI >= 30.0)'))
        sleep_time = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)
        gen_health = st.sidebar.selectbox("How can you define your general health?",
                                          options=('Very good' ,'Fair', 'Good', 'Poor', 'Excellent'))
        phys_health = st.sidebar.number_input("For how many days during the past 30 days was"
                                              " your physical health not good?", 0, 30, 0)
        ment_health = st.sidebar.number_input("For how many days during the past 30 days was"
                                              " your mental health not good?", 0, 30, 0)
        phys_act = st.sidebar.selectbox("Have you played any sports (running, biking, etc.)"
                                        " in the past month?", options=("No", "Yes"))
        smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in"
                                       " your entire life (approx. 5 packs)?)",
                                       options=("No", "Yes"))
        alcohol_drink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men)"
                                             " or more than 7 (women) in a week?", options=("No", "Yes"))
        stroke = st.sidebar.selectbox("Did you have a stroke?", options=("No", "Yes"))
        diff_walk = st.sidebar.selectbox("Do you have serious difficulty walking"
                                         " or climbing stairs?", options=("No", "Yes"))
        diabetic = st.sidebar.selectbox("Have you ever had diabetes?",
                                        options=('Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'))
        asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
        kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
        skin_canc = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))

        features = pd.DataFrame({
            "BMI": [bmi_cat],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [phys_act],
            "GenHealth": [gen_health],
            "SleepTime": [sleep_time],
            "Asthma": [asthma],
            "KidneyDisease": [kid_dis],
            "SkinCancer": [skin_canc]
        })

        return features

    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="images/heart-fav.png"
    )

    st.title("Heart Disease Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="I'll help you diagnose your heart health! - Dr. Sigma",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a logistic regression model using an oversampling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        The Dataset is taken from Kaggle and you can make you own application too using this survey.
        We used Random forest which gave(95%, accuracy) but as our data contains 
        mostly binary features we can also use using logistic regression with 82%, accuracy which is also good.
        
        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.**
        """)

    

    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

    data = user_input_features()
    
    data['SkinCancer']=data['SkinCancer'].replace(['No','Yes'],[0,1])
    data['Asthma']=data['Asthma'].replace(['No','Yes'],[0,1])
    data['KidneyDisease']=data['KidneyDisease'].replace(['No','Yes'],[0,1])
    data['Stroke']=data['Stroke'].replace(['No','Yes'],[0,1])
    data['DiffWalking']=data['DiffWalking'].replace(['No','Yes'],[0,1])
    data['Smoking']=data['Smoking'].replace(['No','Yes'],[0,1])
    data['AlcoholDrinking']=data['AlcoholDrinking'].replace(['No','Yes'],[0,1])
    data['Sex']=data['Sex'].replace(['Female','Male'],[0,1])
    data['PhysicalActivity']=data['PhysicalActivity'].replace(['No','Yes'],[0,1])
    data['GenHealth']=data['GenHealth'].replace(['Very good' ,'Fair', 'Good', 'Poor', 'Excellent'],[2,4,3,5,1])
    data['Race']=data['Race'].replace(['White', 'Black','Asian', 'American Indian/Alaskan Native','Other','Hispanic'],[1,3,5,6,4,2])
    data['Diabetic']=data['Diabetic'].replace(['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'],[2,1,3,4])

    d={'18-24': 21,
    '25-29': 27,
    '30-34': 32,
    '35-39': 37,
    '40-44': 42,
    '45-49': 47,
    '50-54': 52,
    '55-59': 57,
    '60-64': 62,
    '65-69': 67,
    '70-74': 72,
    '75-79': 77,
    '80 or older': 82}
    data['AgeCategory']=data['AgeCategory'].map(d)
    data['BMI_Normal']=np.where(data['BMI']=='Normal(BMI > 18.5 and <25.5)',1,0)
    data['BMI_Underweight']=np.where(data['BMI']=='Underweight (BMI <= 18.5)',1,0)
    data['BMI_Overweight']=np.where(data['BMI']=='Overweight(BMI >=25.5 and BMI<30.0)',1,0)
    data['BMI_Obese']=np.where(data['BMI']=='Obese (BMI >= 30.0)',1,0)
    data.drop('BMI',axis=1,inplace=True)

    data=data.reindex(columns=['Smoking','AlcoholDrinking',
    'Stroke','PhysicalHealth','MentalHealth',
    'DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity',
    'GenHealth','SleepTime','Asthma','KidneyDisease','SkinCancer',
    'BMI_Normal','BMI_Underweight','BMI_Overweight','BMI_Obese'])


    data.fillna(0, inplace=True)

    log_model= open(LOG_MODEL_PATH,"rb")
    log_model=pickle.load(log_model)

    if submit:
        prediction = log_model.predict(data)
        prediction_prob = log_model.predict_proba(data)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. Sigma")
        else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. Sigma")


if __name__ == "__main__":
    main()
