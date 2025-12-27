import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Lets load the joblib instances over here
with open('pipeline.joblib','rb') as file:
    preprocess = joblib.load(file)
    
with open('model.joblib','rb') as file:
    model = joblib.load(file)
    

st.title('HELP NGO Organization')
st.subheader('This application will help to identify the development category of a country using socio-economic factors. Original data has been clustered using KMeans.')

# Lets take inputs from the user
gdpp = st.number_input('Enter GDP per capita (Calculated as the Total GDP divided by the total population):')
income = st.number_input('Enter Income per population:')
imports = st.number_input('Enter Imports of goods and services per capita:')
exports = st.number_input('Enter Exports of goods and services per capita:')
inflation=st.number_input('Enter Inflation (The measurement of the annual growth rate of the Total GDP):')
lf_expecy=st.number_input('Enter Life Expectancy (The average number of years a new born child would live if the current mortality patterns are to remain the same):')
fert = st.number_input('Enter Fertility (The number of children that would be born to each woman if the current age):')
health = st.number_input('Enter Total Health Spending per capita:')
child_mort = st.number_input('Enter Child Mortality (Death of children under 5 years of age per 1000 live births):')

input_list = [child_mort,exports,health,imports,income,inflation,lf_expecy,fert,gdpp]

final_input_list = preprocess.transform([input_list])

if st.button('Predict'):
    prediction = model.predict(final_input_list)[0]
    if prediction == 0:
        st.success('Developing')
    elif prediction == 1:
        st.success('Developed')
    else:
        st.error('Underdeveloped')