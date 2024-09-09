import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.time_series import load_model, predict_model

# Sidebar information
st.sidebar.header('About this Model')
st.sidebar.markdown("""
This app uses Time series models to make predictions and displays the results automatically.
Select a company and year to view forecasts and historical data.
""")

# Define model paths
model_paths = {
    "Meta Scope 1": 'model/meta_scope1_model',
    "Meta Scope 2": 'model/meta_scope2_model',
    "Meta Scope 3": 'model/meta_scope3_model',
    "Fujitsu Scope 1": 'model/fujitsu_scope1_model',
    "Fujitsu Scope 2": 'model/fujitsu_scope2_model',
    "Fujitsu Scope 3": 'model/fujitsu_scope3_model',
    "Amazon Scope 1": 'model/amazon_scope1_model',
    "Amazon Scope 2": 'model/amazon_scope2_model',
    "Amazon Scope 3": 'model/amazon_scope3_model',
    "Google Scope 1": 'model/google_scope1_model',
    "Google Scope 2": 'model/google_scope2_model',
    "Google Scope 3": 'model/google_scope3_model',
    "Microsoft Scope 1": 'model/microsoft_scope1_model',
    "Microsoft Scope 2": 'model/microsoft_scope2_model',
    "Microsoft Scope 3": 'model/microsoft_scope3_model'
}

# Define historical data paths
historical_data_paths = {
    "Meta Scope 1": 'data/meta_scope1.csv',
    "Meta Scope 2": 'data/meta_scope2.csv',
    "Meta Scope 3": 'data/meta_scope3.csv',
    "Fujitsu Scope 1": 'data/fujitsu_scope1.csv',
    "Fujitsu Scope 2": 'data/fujitsu_scope2.csv',
    "Fujitsu Scope 3": 'data/fujitsu_scope3.csv',
    "Amazon Scope 1": 'data/amazon_scope1.csv',
    "Amazon Scope 2": 'data/amazon_scope2.csv',
    "Amazon Scope 3": 'data/amazon_scope3.csv',
    "Google Scope 1": 'data/google_scope1.csv',
    "Google Scope 2": 'data/google_scope2.csv',
    "Google Scope 3": 'data/google_scope3.csv',
    "Microsoft Scope 1": 'data/microsoft_scope1.csv',
    "Microsoft Scope 2": 'data/microsoft_scope2.csv',
    "Microsoft Scope 3": 'data/microsoft_scope3.csv'
}

# Load models
def load_models(model_paths):
    models = {}
    for model_name, model_path in model_paths.items():
        try:
            models[model_name] = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model '{model_name}': {e}")
    return models

# Load historical data
def load_historical_data(data_paths):
    data = {}
    for model_name, path in data_paths.items():
        try:
            data[model_name] = pd.read_csv(path, index_col='Year', parse_dates=True)
        except Exception as e:
            st.error(f"Error loading historical data for '{model_name}': {e}")
    return data

# Load models and historical data
models = load_models(model_paths)
historical_data = load_historical_data(historical_data_paths)

# Function to combine historical and prediction data
def combine_data(historical, prediction, label):
    # Ensure the prediction length matches the forecast horizon
    pred_index = pd.date_range(start=historical.index[-1] + pd.DateOffset(1), periods=len(prediction), freq='Y')
    prediction_series = pd.Series(prediction, index=pred_index, name=f'Prediction {label}')
    
    # Combine the historical data with predictions
    combined = pd.concat([historical, prediction_series], axis=0)
    combined.columns = [f'{label} Original', f'{label} Prediction']
    return combined

# Streamlit App
st.title('PyCaret Time Series Carbon Emission Forecasts')

# File uploader for user CSV input
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load user-uploaded data if provided
user_data = None
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file, index_col='Year', parse_dates=True)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# User input for company and year
company = st.sidebar.selectbox('Select a company:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], index=0)
year = st.sidebar.slider('Select year:', min_value=2017, max_value=2025, value=2024)

# Tabs for Combined Charts and Individual Scope Charts
tab1, tab2 = st.tabs(["Combined Charts", "Individual Scope Charts"])

# Get the relevant model names for the selected company
model_names = [f"{company} Scope 1", f"{company} Scope 2", f"{company} Scope 3"]

combined_data_list = []

for scope in model_names:
    if scope in models:
        predictions = predict_model(models[scope], fh=30)
        combined_data = combine_data(historical_data[scope], predictions.values.flatten(), scope)
        combined_data_list.append(combined_data)

# Combine all scopes into a single DataFrame for plotting
final_combined_data = pd.concat(combined_data_list, axis=1)

# Add user data to the charts if available
if user_data is not None:
    #user_data['Year'] = pd.to_datetime(user_data['Year'])
    user_data.set_index('Year', inplace=True)
    # Rename columns to align with the chart data
    user_data.columns = ['User Scope 1', 'User Scope 2', 'User Scope 3']
    final_combined_data = pd.concat([final_combined_data, user_data], axis=1)

# Combined Charts Tab
with tab1:
    st.subheader(f'{company} Carbon Emissions: Scopes 1, 2, and 3 (Original vs Predictions)')
    fig_combined = px.line(final_combined_data, 
                           x=final_combined_data.index, 
                           y=final_combined_data.columns, 
                           title=f'{company} Carbon Emissions: Scopes 1, 2, and 3 (Original vs Predictions)', 
                           labels={"index": "Year", "value": "Emissions (in metric tons)"})
    st.plotly_chart(fig_combined)

# Individual Scope Charts Tab
with tab2:
    for scope in model_names:
        st.subheader(f'{company} {scope} (Original vs Prediction)')
        fig_scope = px.line(final_combined_data[[f'{scope} Original', f'{scope} Prediction']], 
                            x=final_combined_data.index, 
                            y=[f'{scope} Original', f'{scope} Prediction'], 
                            title=f'{company} {scope} (Original vs Prediction)', 
                            labels={"index": "Year", "value": "Emissions (in metric tons)"})
        st.plotly_chart(fig_scope)
    
    # Add User Data Chart if available
    if user_data is not None:
        st.subheader('User Uploaded Data (Scope 1, Scope 2, Scope 3)')
        fig_user = px.line(user_data, 
                           x=user_data.index, 
                           y=user_data.columns, 
                           title='User Uploaded Data (Scope 1, Scope 2, Scope 3)', 
                           labels={"index": "Year", "value": "Emissions (in metric tons)"})
        st.plotly_chart(fig_user)
