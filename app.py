import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model, scaler, encoder, and feature column names
@st.cache_resource
def load_model_components():
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('robust_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('onehot_encoder.pkl', 'rb') as file:
        ohe = pickle.load(file)
    with open('feature_columns.pkl', 'rb') as file:
        trained_feature_columns = pickle.load(file)
    return model, scaler, ohe, trained_feature_columns

model, scaler, ohe, trained_feature_columns = load_model_components()

# Define nominal columns based on your training data
nominal_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]

# Title of the Streamlit app
st.title('House Price Prediction App')
st.write('Enter the details of the house to predict its price.')

# Input fields for numerical features
area = st.number_input('Area (in sqft)', min_value=100, max_value=20000, value=3000)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=5, value=2)
stories = st.number_input('Number of Stories', min_value=1, max_value=5, value=2)
parking = st.number_input('Number of Parking spaces', min_value=0, max_value=5, value=1)

# Input fields for categorical features
mainroad = st.selectbox('Main Road Access', ['yes', 'no'])
guestroom = st.selectbox('Guest Room', ['yes', 'no'])
basement = st.selectbox('Basement', ['yes', 'no'])
hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'])
airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'])
prefarea = st.selectbox('Preferred Area', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

# Preprocessing the input data
def preprocess_input(df_input, ohe_model, scaler_model, nominal_columns, trained_feature_columns):
    # Separate numerical and nominal columns from input_data
    numerical_cols_input = [col for col in df_input.columns if col not in nominal_columns]
    
    # Apply OneHotEncoder to nominal columns
    encoded_nominal_data = ohe_model.transform(df_input[nominal_columns])
    encoded_nominal_df = pd.DataFrame(encoded_nominal_data, columns=ohe_model.get_feature_names_out(nominal_columns))

    # Combine numerical and encoded nominal features
    df_processed = pd.concat([df_input[numerical_cols_input].reset_index(drop=True), encoded_nominal_df.reset_index(drop=True)], axis=1)

    # Ensure all trained columns are present, fill missing with 0, and reorder
    for col in trained_feature_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0 # This handles cases where a category was in training but not in current input
    df_processed = df_processed[trained_feature_columns] # Reorder columns to match training

    # Apply RobustScaler
    scaled_data = scaler_model.transform(df_processed)
    return scaled_data

if st.button('Predict Price'):
    processed_input = preprocess_input(input_data.copy(), ohe, scaler, nominal_cols, trained_feature_columns)
    prediction = model.predict(processed_input)[0]
    st.success(f'The predicted house price is: ₹{prediction:,.2f}')
