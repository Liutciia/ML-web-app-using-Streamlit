import joblib
from pickle import load
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# Función para cargar el modelo
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

# Cargar el modelo
model_path = 'decision_tree_classifier_default_42.sav'
model = load_model(model_path)

class_dict = {
    "0": "Diabetes negative",
    "1": "Diabetes positive"
}
st.write("""
# Diabetes Prediction App
This app predicts **Diabetes**!
""")

val1 = st.slider("Glucose", min_value = 0, max_value = 199, step = 10)
val2 = st.slider("BloodPressure", min_value = 0, max_value = 122, step = 10)
val3 = st.slider("Insulin", min_value = 0, max_value = 846, step = 10)
val4 = st.slider("BMI", min_value = 0.0, max_value = 67.10, step = 1.0)
val5 = st.slider("DiabetesPedigreeFunction", min_value = 0.078, max_value = 2.42, step = 0.1)
val6 = st.slider("Age", min_value = 21, max_value = 81, step = 1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)