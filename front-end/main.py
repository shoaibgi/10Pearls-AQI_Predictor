import streamlit as st
import pickle
import joblib
import sklearn
import os
import pandas as pd
from tensorflow import keras
import tensorflow as tf

image='10Pearls_Logo.jpg'
st.image(image,width=200,)
st.title('10Pearls-AQI Predictor')
st.info('This web app uses a machine learning model to \
          predict the Air Quality Index (AQI) of a city \
          based on the input features such as \
          T, TM, SLP, H, VV, V, VM')

with st.expander('Data'):
 st.write('**AQI-Data**')
 df=pd.read_csv('Real_combine.csv')
 df

 st.write('**X**')
 X=df.iloc[:,:-1]
 X

 st.write('**y**')
 y=df.iloc[:,-1]
 y

with st.expander('Feature Visualization'):
     st.write('**Features-Scatter_Plot**')
     st.scatter_chart(data=df, x='T', y='VM', color='SLP')

# ---------------- Model Selection ---------------- #
model_names = ['random_forest_regression_model', 'decision_regression_model', 'XGTBoost-Regressor_model']
model_selection = st.selectbox('sklearn Model', model_names)

tensor = ['NN_model']
tensor_selection = st.selectbox('TensorFlow Model', tensor)

if model_selection == 'random_forest_regression_model':
    model_path = "D:/10Pearls-AQI/random_forest_regression_model.pkl"
elif model_selection == 'decision_regression_model':
    model_path = "D:/10Pearls-AQI/decision_regression_model.pkl"
elif model_selection == 'XGTBoost-Regressor_model':
    model_path = "D:/10Pearls-AQI/XGTBoost-Regressor_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

tensor_path = "D:/10Pearls-AQI/NN_model.h5"
tensor_model = tf.keras.models.load_model(tensor_path)

# ---------------- Sidebar Inputs ---------------- #
st.sidebar.title('Input Parameters')

T = st.sidebar.number_input('T', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
TM = st.sidebar.number_input('TM', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
Tm = st.sidebar.number_input('Tm', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
SLP = st.sidebar.number_input('SLP', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
H = st.sidebar.number_input('H', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
VV = st.sidebar.number_input('VV', min_value=0.0, max_value=40.0, value=0.5, step=0.5)
V = st.sidebar.number_input('V', min_value=0.0, max_value=40.0, value=0.5, step=0.5)
VM = st.sidebar.number_input('VM', min_value=0.0, max_value=1000.0, value=25.0, step=10.0)

input_data = {
    'T': [T],
    'TM': [TM],
    'Tm': [Tm],
    'SLP': [SLP],
    'H': [H],
    'VV': [VV],
    'V': [V],
    'VM': [VM]
}

input_df = pd.DataFrame(input_data)
required_features = ["T", "TM", "Tm", "SLP", "H", "VV", "V", "VM"]
for col in required_features:
    if col not in input_df.columns:
        input_df[col] = 0  # default value if missing

input_df = input_df[required_features]   # ensure correct order

# Tensor data (for NN model)
tensor_data = tf.constant([[T, TM, Tm, SLP, H, VV, V, VM]], dtype=tf.float32)

# ---------------- Prediction Section ---------------- #
col1, col2 = st.columns([1, 1])
col1.subheader('Click on the Button to predict sklearn-model')

if st.button("Predict AQI"):
    prediction = model.predict(input_df)

    # Convert numpy array → scalar
    prediction_value = float(prediction[0])

    # Define color ranges
    color_ranges = {
        (0, 50): '#1FE140',
        (51, 100): '#F5B700',
        (101, 150): '#F26430',
        (151, 200): '#DF2935',
        (201, 300): '#D77A61',
        (301, float('inf')): '#4D5061'
    }

    aqi_quality_table = {
        (0, 50): 'Good',
        (51, 100): 'Satisfactory',
        (101, 150): 'Moderate',
        (151, 200): 'Poor',
        (201, 300): 'Very Poor',
        (301, float('inf')): 'Severe'
    }

    # Find AQI category
    prediction_color = 'black'
    aqi_quality = ''
    for range_, color in color_ranges.items():
        if range_[0] <= prediction_value <= range_[1]:
            prediction_color = color
            aqi_quality = aqi_quality_table[range_]
            break

    # Display styled result
    st.markdown(f"""
    <div style="background-color:{prediction_color}; padding:10px; border-radius:5px;">
        <p style="color:white; font-size:25px;">{aqi_quality}</p>
        <p style="color:white; font-size:25px;">Predicted AQI: {prediction_value:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"Predicted AQI: **{prediction_value:.2f}**")

# ---------------- TensorFlow Prediction ---------------- #
col1, col2 = st.columns([1, 1])
col1.subheader('Click on the Button to predict TensorFlow-model ')

if st.button("Tensor-Predict AQI"):
    tensor_prediction = tensor_model.predict(tensor_data)
    tensor_value = float(tensor_prediction.item())

    tensor_ranges = {
        (0, 50): '#1FE140',
        (51, 100): '#F5B700',
        (101, 150): '#F26430',
        (151, 200): '#DF2935',
        (201, 300): '#D77A61',
        (301, float('inf')): '#4D5061'
    }

    tensor_aqi_quality_table = {
        (0, 50): 'Good',
        (51, 100): 'Satisfactory',
        (101, 150): 'Moderate',
        (151, 200): 'Poor',
        (201, 300): 'Very Poor',
        (301, float('inf')): 'Severe'
    }

    # Find AQI category
    tensor_prediction_color = 'black'
    tensor_aqi_quality = ''
    for range_, color in tensor_ranges.items():
        if range_[0] <= tensor_value <= range_[1]:
            tensor_prediction_color = color
            tensor_aqi_quality = tensor_aqi_quality_table[range_]
            break

    # Display styled result
    st.markdown(f"""
    <div style="background-color:{tensor_prediction_color}; padding:10px; border-radius:5px;">
        <p style="color:white; font-size:25px;">{tensor_aqi_quality}</p>
        <p style="color:white; font-size:25px;">Predicted AQI: {tensor_value:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    st.success(f"Tensor-AQI: **{tensor_value:.2f}**")

# ---------------- Image Section ---------------- #
image_folder = 'images'
image_files = sorted(os.listdir(image_folder))

st.subheader("Training Results")
image_index = st.slider("Slide Images",0, len(image_files) - 1  , 0)

image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split('.')[0].replace('_', ' ').title()

st.title(image_title)
st.image(image_path, use_container_width=True)
