import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.time_series import TSForecastingExperiment

# Sidebar information
st.sidebar.header('About this Model')
st.sidebar.markdown("""
This app uses Time series models to make predictions and displays the results automatically.
Select a company and year to view forecasts and historical data. If you upload your own CSV file, the app will train a model on that data and include the predictions in the charts.
""")

# Initialise file name
file_name = ""

# Define historical data paths (same as before)

# Load historical data (same as before)

# Function to combine historical and prediction data (same as before)

# Streamlit App
st.title('Time Series Carbon Emission Forecasts')

# File uploader for user CSV input
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Process user-uploaded file if provided
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file, index_col='Year', parse_dates=True)
        file_name = uploaded_file.name.split('_')[0]
        st.sidebar.success("File uploaded successfully!")

        # Let the user select the target column
        target_column = st.sidebar.selectbox('Select the target column for forecasting:', user_data.columns)

        # PyCaret Time Series Experiment
        exp = TSForecastingExperiment()
        exp.setup(user_data[target_column], fh=30, fold=3, session_id=123)

        # Train a model (auto_arima in this case)
        model = exp.create_model('auto_arima')

        # Make predictions
        predictions = exp.predict_model(model)

        # Combine user-uploaded historical data with predictions
        user_combined_data = combine_data(user_data[target_column], predictions, file_name)

    except Exception as e:
        st.sidebar.error(f"Error loading file or training model: {e}")
        user_data = None

# Tabs for Combined Charts, Individual Scope Charts, and Data Table
tab1, tab2, tab3 = st.tabs(["Combined Charts", "Individual Scope Charts", "Emission Data Table"])

# Combined Charts Tab
with tab1:
    st.subheader(f'{file_name} Carbon Emissions: Scopes 1, 2, and 3 (Original vs Predictions)')
    
    # If user data is available, add it to the chart
    if user_data is not None:
        fig_user = px.line(user_combined_data, 
                           x=user_combined_data.index, 
                           y=user_combined_data.columns, 
                           title=f'{file_name} (Original vs Predictions)', 
                           labels={"index": "Year", "value": "Emissions (in metric tons)"})
        st.plotly_chart(fig_user)

# Individual Scope Charts Tab
with tab2:
    st.subheader(f'{file_name} Scope 1, 2, and 3 (Original vs Predictions)')
    if user_data is not None:
        st.subheader(f'{file_name} (Original vs Predictions)')
        fig_user_scope = px.line(user_combined_data,
                                 x=user_combined_data.index,
                                 y=user_combined_data.columns,
                                 title=f'{file_name} (Original vs Predictions)',
                                 labels={"index": "Year", "value": "Emissions (in metric tons)"})
        st.plotly_chart(fig_user_scope)

# Data Table Tab
with tab3:
    # Show the table with the user's original data and predictions
    if user_data is not None:
        st.subheader(f'💨 Carbon Emissions Table for {file_name}')
        st.write(user_combined_data)

# Add Download Button for User Data Predictions
if user_data is not None:
    csv = user_combined_data.to_csv().encode('utf-8')
    st.download_button(label="Download User Data as CSV", data=csv, file_name=f'{file_name}_emissions_predictions.csv', mime='text/csv')
