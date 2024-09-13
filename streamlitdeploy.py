import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.time_series import TSForecastingExperiment

# Sidebar information
st.sidebar.header('About this Model')
st.sidebar.markdown("""
This app uses Time series models to make predictions and displays the results automatically.
Upload your own CSV file, and the app will train models for Scope 1, Scope 2, and Scope 3 data, then include the predictions in the charts.
""")

# Initialise file name
file_name = ""

# Define function to combine historical and prediction data
def combine_data(historical, prediction, label):
    pred_index = pd.date_range(start=historical.index[-1] + pd.DateOffset(1), periods=len(prediction), freq='Y')
    prediction_series = pd.Series(prediction, index=pred_index, name=f'Prediction {label}')
    combined = pd.concat([historical, prediction_series], axis=0)
    combined.columns = [f'{label} Original', f'{label} Prediction']
    return combined

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
        
        # Check if the file contains Scope 1, Scope 2, and Scope 3 columns
        if all(scope in user_data.columns for scope in ['Scope 1', 'Scope 2', 'Scope 3']):
            
            # Dictionary to store combined data for each scope
            combined_data_dict = {}

            # PyCaret Time Series Experiment
            exp = TSForecastingExperiment()

            for scope in ['Scope 1', 'Scope 2', 'Scope 3']:
                # Get the length of the data to determine a suitable forecast horizon
                data_length = len(user_data[scope])

                # Set the forecast horizon to be smaller than the data length if necessary
                fh = min(10, data_length // 2)  # Forecast up to 10 periods, but not more than half of the data length

                if data_length < 6:
                    st.error(f"Not enough data points in {scope} for model training. At least 6 data points are required.")
                else:
                    # Setup PyCaret for each scope
                    exp.setup(user_data[scope], fh=fh, fold=3, session_id=123)

                    # Train a model (auto_arima in this case)
                    model = exp.create_model('auto_arima')

                    # Make predictions
                    predictions = exp.predict_model(model)

                    # Combine user-uploaded historical data with predictions for the current scope
                    combined_data_dict[scope] = combine_data(user_data[scope], predictions, f'{file_name} {scope}')

        else:
            st.sidebar.error("Uploaded file must contain 'Scope 1', 'Scope 2', and 'Scope 3' columns.")
            user_data = None

    except Exception as e:
        st.sidebar.error(f"Error loading file or training model: {e}")
        user_data = None

# Tabs for Combined Charts, Individual Scope Charts, and Data Table
if user_data is not None:
    tab1, tab2, tab3 = st.tabs(["Combined Charts", "Individual Scope Charts", "Emission Data Table"])

    # Combine the results for all scopes into a single DataFrame for plotting
    combined_all_scopes = pd.concat([combined_data_dict['Scope 1'], combined_data_dict['Scope 2'], combined_data_dict['Scope 3']], axis=1)

    # Combined Charts Tab
    with tab1:
        st.subheader(f'{file_name} Carbon Emissions: Scopes 1, 2, and 3 (Original vs Predictions)')
        fig_combined = px.line(combined_all_scopes, 
                               x=combined_all_scopes.index, 
                               y=combined_all_scopes.columns, 
                               title=f'{file_name} Carbon Emissions: Combined Scopes 1, 2, and 3',
                               labels={"index": "Year", "value": "Emissions (in metric tons)"})
        st.plotly_chart(fig_combined)

    # Individual Scope Charts Tab
    with tab2:
        for scope in ['Scope 1', 'Scope 2', 'Scope 3']:
            st.subheader(f'{file_name} {scope} (Original vs Predictions)')
            fig_scope = px.line(combined_data_dict[scope], 
                                x=combined_data_dict[scope].index, 
                                y=combined_data_dict[scope].columns, 
                                title=f'{file_name} {scope} (Original vs Predictions)',
                                labels={"index": "Year", "value": "Emissions (in metric tons)"})
            st.plotly_chart(fig_scope)

    # Data Table Tab
    with tab3:
        st.subheader(f'💨 Carbon Emissions Table for {file_name}')
        st.write(combined_all_scopes)

    # Add Download Button for User Data Predictions
    csv = combined_all_scopes.to_csv().encode('utf-8')
    st.download_button(label="Download Data as CSV", data=csv, file_name=f'{file_name}_emissions_predictions.csv', mime='text/csv')
