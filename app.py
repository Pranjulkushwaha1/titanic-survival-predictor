import streamlit as st
import pickle
import numpy as np 
import pandas as pd 

with open('titanic_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('titanic_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Titanic Survival Prediction')
st.write('Passenger ki details dalo aur pata karo ki wo bachega ya nahi!')

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age", min_value=0, max_value=80, value=25)
sibsp = st.number_input("Siblings/Spouse", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children",min_value=0, max_value=9, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=512.0, value=32.0)
embarked = st.selectbox("Embarked", ['S','C','Q'])

if sex == 'male' and age < 18:
    title = 'Master'
elif sex == 'male' and age >= 18:
    title = 'Mr'
elif sex == 'female' and age < 18:
    title = 'Miss'
else:
    title = 'Mrs'

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0


# Predict button
if st.button("Predict Survival"):
    
    # Encoding
    sex_female = 1 if sex == 'female' else 0
    sex_male = 1 if sex == 'male' else 0
    
    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0
    
    title_Master = 1 if title == 'Master' else 0
    title_Miss = 1 if title == 'Miss' else 0
    title_Mr = 1 if title == 'Mr' else 0
    title_Mrs = 1 if title == 'Mrs' else 0
    title_Other = 0

    input_data = np.array([[
        pclass, age, sibsp, parch, fare,
        family_size, is_alone,
        sex_female, sex_male,
        embarked_C, embarked_Q, embarked_S,
        title_Master, title_Miss, title_Mr, title_Mrs, title_Other
    ]])
    
    input_scaled = scaler.transform(input_data)
    st.write(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("✅ Passenger Survived!")
    else:
        st.error("❌ Passenger Did Not Survive!")

