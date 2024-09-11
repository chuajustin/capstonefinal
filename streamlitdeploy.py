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

# Initialise file name
file_name = ""

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
st.title('Time Series Carbon Emission Forecasts')
# User input for company and year
company = st.sidebar.selectbox('Select a company:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], index=0)

# File uploader for user CSV input
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load user-uploaded data if provided
user_data = None
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file, index_col='Year', parse_dates=True)
        file_name = uploaded_file.name.split('_')[0]
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Tabs for Combined Charts, Individual Scope Charts, and Data Table
tab1, tab2, tab3 = st.tabs(["Combined Charts", "Individual Scope Charts", "Emission Data Table"])

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
    # Rename columns to align with the chart data
    user_data.columns = [f'{file_name} Scope 1 Original', f'{file_name} Scope 2 Original', f'{file_name} Scope 3 Original']
    final_combined_data = pd.concat([final_combined_data, user_data], axis=1)


# Combined Charts Tab
with tab1:
    st.subheader(f'{company} Carbon Emissions: Scopes 1, 2, and 3 (Original vs Predictions)')
    # Multi-select widget to choose companies for comparison
    companies_to_compare = st.multiselect('Compare with:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], key='company_comparison')

    
    if companies_to_compare:
        combined_data_list = []

        for company in companies_to_compare:
            model_names = [f"{company} Scope 1", f"{company} Scope 2", f"{company} Scope 3"]
            for scope in model_names:
                if scope in models:
                    predictions = predict_model(models[scope], fh=30)
                    combined_data = combine_data(historical_data[scope], predictions.values.flatten(), scope)
                    combined_data_list.append(combined_data)

        if combined_data_list:
            final_combined_data = pd.concat(combined_data_list, axis=1)

        # Render a line chart with the combined data
        fig_combined = px.line(final_combined_data, 
                               x=final_combined_data.index, 
                               y=final_combined_data.columns, 
                               title=f'{company}: Comparing Scopes 1, 2, and 3 with Selected Companies', 
                               labels={"index": "Year", "value": "Emissions (in metric tons)"})

    else: 
        fig_combined = px.line(final_combined_data, 
                               x=final_combined_data.index, 
                               y=final_combined_data.columns, 
                               title=f'Compare against {file_name} Original', 
                               labels={"index": "Year", "value": "Emissions (in metric tons)"})



    #Display the chart with annotations
    st.plotly_chart(fig_combined)

# Individual Scope Chart
with tab2:
    # Multi-select widget to choose companies for comparison
    companies_to_compare = st.multiselect('Compare with:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], key='company_comparison_indv')

    if companies_to_compare:
        st.subheader('Comparison of Selected Companies')
        
        # Loop through each scope (Scope 1, Scope 2, Scope 3)
        for scope in ['Scope 1', 'Scope 2', 'Scope 3']:
            # Initialize an empty DataFrame to hold the comparison data for the current scope
            comparison_data = pd.DataFrame()

            # Collect data for each selected company for the current scope
            for comp in companies_to_compare:
                scope_name = f"{comp} {scope}"
                if scope_name in models:
                    try:
                        # Make predictions
                        predictions = predict_model(models[scope_name], fh=30)
                        st.write(predictions.head())  # Display the first few rows to understand its structure
                        # Combine historical and predicted data
                        combined_data = combine_data(historical_data[scope_name], predictions.values.flatten(), f'{comp} {scope}')
                        comparison_data[f'{comp} {scope} Original'] = combined_data[f'{comp} {scope} Original']
                        comparison_data[f'{comp} {scope} Prediction'] = combined_data[f'{comp} {scope} Prediction']
                    except Exception as e:
                        st.error(f"Error with {scope_name}: {e}")

            # Plot the comparison data for the current scope if any data exists
            if not comparison_data.empty:
                st.subheader(f'{scope} Comparison (Original vs Predictions)')

                # Create two columns: one for the chart, one for the forecast values
                col1, col2 = st.columns([2, 1])

                # In the first column, display the chart
                with col1:
                    fig_scope_compare = px.line(comparison_data,
                                                x=comparison_data.index,
                                                y=comparison_data.columns,
                                                title=f'{scope} Comparison: Original vs Predictions',
                                                labels={"index": "Year", "value": "Emissions (in metric tons)"})
                    st.plotly_chart(fig_scope_compare)

                # In the second column, display the forecast values for 2030 and 2050
                with col2:
                    try:
                        forecast_2030 = y_pred.loc['2030'] if '2030' in comparison_data.index else "2030 data not available"
                        forecast_2050 = y_pred.loc['2050'] if '2050' in comparison_data.index else "2050 data not available"
                        st.write(f"### {scope} Forecast")
                        st.write(f"- **2030 Forecast**: {forecast_2030}")
                        st.write(f"- **2050 Forecast**: {forecast_2050}")
                    except Exception as e:
                        st.write("Error fetching forecast data:", e)

            else:
                st.warning(f"No data available for {scope} comparison.")

    else:
        for scope in model_names:
            st.subheader(f'{company} {scope} (Original vs Prediction)')

            if f'{scope} Original' in final_combined_data.columns and f'{scope} Prediction' in final_combined_data.columns:
                # Create two columns: one for the chart, one for the forecast values
                col1, col2 = st.columns([2, 1])

                # In the first column, display the chart
                with col1:
                    fig_scope = px.line(final_combined_data[[f'{scope} Original', f'{scope} Prediction']],
                                        x=final_combined_data.index,
                                        y=[f'{scope} Original', f'{scope} Prediction'],
                                        title=f'{company} {scope} (Original vs Prediction)',
                                        labels={"index": "Year", "value": "Emissions (in metric tons)"})
                    st.plotly_chart(fig_scope)

                # In the second column, display the forecast values for 2030 and 2050
                with col2:
                    forecast_2030 = final_combined_data.loc[2030] if 2030 in final_combined_data.index else "2030 data not available"
                    forecast_2050 = final_combined_data.loc[2050] if 2050 in final_combined_data.index else "2050 data not available"
                    st.write(f"### {scope} Forecast")
                    st.write(f"- **2030 Forecast**: {forecast_2030}")
                    st.write(f"- **2050 Forecast**: {forecast_2050}")


# Data Table Tab
with tab3:
# Create the subheader with conditional inclusion
    if uploaded_file:
        subheader_text = f'💨 Carbon Emissions Table including {file_name}'
    else:
        subheader_text = '💨 Carbon Emissions Table'
    st.subheader(subheader_text)

    carbon_emissions_table = final_combined_data
    
    # Convert the index to datetime if it's not already in datetime format
    carbon_emissions_table.index = pd.to_datetime(carbon_emissions_table.index)
        
    # Extract the first 4 characters (year) from the index and assign it back
    carbon_emissions_table.index = carbon_emissions_table.index.strftime('%Y')
        
    # Rename the index to 'Year'
    carbon_emissions_table.index.name = 'Year'
    
    # Display the updated table
    st.write(carbon_emissions_table)



# Download as CSV
csv = final_combined_data.to_csv().encode('utf-8')
st.download_button(label="Download data as CSV", data=csv, file_name=f'{company}_emissions_comparison.csv', mime='text/csv')
