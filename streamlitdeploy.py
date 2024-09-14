import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.time_series import load_model, predict_model

# Sidebar information
st.sidebar.header('About this Model')
st.sidebar.markdown("""
This app uses Time series models to make predictions and displays the results automatically. Historical Data contains Scope 1, 2 and 3 emissions.

Select a company to view forecasts and historical data.
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
st.title('''Carbon Cast 💨
A Time Series Carbon Emission Forecast by Justin''')
# User input for company and year
company = st.sidebar.selectbox('Select a company:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], index=0)


# File uploader for user CSV input
uploaded_file = st.sidebar.file_uploader("Upload your CSV file here for comparison and prediction", type=["csv"])

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

    # Scope 1 - Initialise a list to store predictions of user uploaded scopes
    predictions_combined_data_list = []

    predictions_user_upload = predict_model(models["Meta Scope 1"], fh=30)
    predictions_combined_data = combine_data(historical_data["Meta Scope 1"], predictions_user_upload.values.flatten(), file_name + " Scope 1")
    predictions_combined_data_list.append(predictions_combined_data)

    # Combine all scopes into a single DataFrame for plotting
    predictions_final_combined_data = pd.concat(predictions_combined_data_list, axis=1)
    
    # Combine user uploaded data with the preloaded data of the 5 companies
    final_combined_data = final_combined_data.join(predictions_final_combined_data.iloc[:,1:2])
    
    
    # Scope 2 - Initialise a list to store predictions of user uploaded scopes
    predictions_combined_data_list = []

    predictions_user_upload = predict_model(models["Meta Scope 2"], fh=30)
    predictions_combined_data = combine_data(historical_data["Meta Scope 2"], predictions_user_upload.values.flatten(), file_name + " Scope 2")
    predictions_combined_data_list.append(predictions_combined_data)

    # Combine all scopes into a single DataFrame for plotting
    predictions_final_combined_data = pd.concat(predictions_combined_data_list, axis=1)
    
    # Combine user uploaded data with the preloaded data of the 5 companies
    final_combined_data = final_combined_data.join(predictions_final_combined_data.iloc[:,1:2])    
    
    
    # Scope 3 - Initialise a list to store predictions of user uploaded scopes
    predictions_combined_data_list = []

    predictions_user_upload = predict_model(models["Meta Scope 3"], fh=30)
    predictions_combined_data = combine_data(historical_data["Meta Scope 3"], predictions_user_upload.values.flatten(), file_name + " Scope 3")
    predictions_combined_data_list.append(predictions_combined_data)

    # Combine all scopes into a single DataFrame for plotting
    predictions_final_combined_data = pd.concat(predictions_combined_data_list, axis=1)
    
    # Combine user uploaded data with the preloaded data of the 5 companies
    final_combined_data = final_combined_data.join(predictions_final_combined_data.iloc[:,1:2])
    
    # Set pandas option to display floats without scientific notation
    pd.set_option('display.float_format', '{:,.3f}'.format)

    # 2030 forecast - for user uploaded data
    forecast_2030_scope1 = str(final_combined_data.iloc[15:16,9:10]).split(" ")[-1]
    forecast_2030_scope2 = str(final_combined_data.iloc[15:16,10:11]).split(" ")[-1]
    forecast_2030_scope3 = str(final_combined_data.iloc[15:16,11:12]).split(" ")[-1]

    st.subheader(f"{file_name} 2030 Forecast in metric tonnes")
    st.write(f"Scope 1: {forecast_2030_scope1}")
    st.write(f"Scope 2: {forecast_2030_scope2}")
    st.write(f"Scope 3: {forecast_2030_scope3}")

    # 2050 forecast - for user uploaded data
    forecast_2050_scope1 = str(final_combined_data.iloc[35:36,9:10]).split(" ")[-1]
    forecast_2050_scope2 = str(final_combined_data.iloc[35:36,10:11]).split(" ")[-1]
    forecast_2050_scope3 = str(final_combined_data.iloc[35:36,11:12]).split(" ")[-1]

    st.subheader(f"{file_name} 2050 Forecast in metric tonnes")
    st.write(f"Scope 1: {forecast_2050_scope1}")
    st.write(f"Scope 2: {forecast_2050_scope2}")
    st.write(f"Scope 3: {forecast_2050_scope3}")

        #st.dataframe(final_combined_data.iloc[35:36,9:])  
    
    
     

# Combined Charts Tab
with tab1:
    st.subheader(f'{company} Carbon Emissions: Scopes 1, 2, and 3 (Original vs Predictions)')
                  # Create two columns: one for the chart, one for the forecast values
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

    # Set pandas option to display floats without scientific notation
    pd.set_option('display.float_format', '{:,.3f}'.format)


    # Ensure index is a DatetimeIndex
    if isinstance(final_combined_data.index, pd.PeriodIndex):
        final_combined_data.index = final_combined_data.index.to_timestamp()
    
    # Retrieve the year from the index
    if isinstance(final_combined_data.index, pd.DatetimeIndex):
        years = final_combined_data.index.year

    # Extract forecasts for 2030 and 2050 dynamically
    try:
        # Construct column names dynamically
        col_scope1 = f"{company} Scope 1 Prediction"
        col_scope2 = f"{company} Scope 2 Prediction"
        col_scope3 = f"{company} Scope 3 Prediction"

     
        # Extract forecasts for 2030
        companies_forecast_2030_scope1 = str(final_combined_data.loc[years == 2030, col_scope1].values[0]).split(" ")[-1]
        companies_forecast_2030_scope2 = str(final_combined_data.loc[years == 2030, col_scope2].values[0]).split(" ")[-1]
        companies_forecast_2030_scope3 = str(final_combined_data.loc[years == 2030, col_scope3].values[0]).split(" ")[-1]

        # Extract forecasts for 2050
        companies_forecast_2050_scope1 = str(final_combined_data.loc[years == 2050, col_scope1].values[0]).split(" ")[-1]
        companies_forecast_2050_scope2 = str(final_combined_data.loc[years == 2050, col_scope2].values[0]).split(" ")[-1]
        companies_forecast_2050_scope3 = str(final_combined_data.loc[years == 2050, col_scope3].values[0]).split(" ")[-1]
    except KeyError as e:
        st.error(f"KeyError: {e}")
        companies_forecast_2030_scope1 = companies_forecast_2030_scope2 = companies_forecast_2030_scope3 = 'Data not available'
        companies_forecast_2050_scope1 = companies_forecast_2050_scope2 = companies_forecast_2050_scope3 = 'Data not available'

    st.subheader(f"{company} 2030 and 2050 Forecast in metric tonnes")
    st.write(f"2030 Scope 1: {companies_forecast_2030_scope1}")
    st.write(f"2030 Scope 2: {companies_forecast_2030_scope2}")
    st.write(f"2030 Scope 3: {companies_forecast_2030_scope3}")
    st.write(f"2050 Scope 1: {companies_forecast_2050_scope1}")
    st.write(f"2050 Scope 2: {companies_forecast_2050_scope2}")
    st.write(f"2050 Scope 3: {companies_forecast_2050_scope3}")


# Comparison case: When users select companies to compare
    if companies_to_compare:
        # Initialize dictionary to store forecasts for each company
        forecast_values = {}
    
        for company in companies_to_compare:
            # Dynamically construct column names for Scope 1, Scope 2, and Scope 3 predictions
            col_scope1 = f"{company} Scope 1 Prediction"
            col_scope2 = f"{company} Scope 2 Prediction"
            col_scope3 = f"{company} Scope 3 Prediction"
    
            # Initialize default values for the company
            forecast_values[company] = {'2030': ['Data not available'] * 3, '2050': ['Data not available'] * 3}
    
            # Check if the columns exist in the DataFrame
            if col_scope1 in final_combined_data.columns and col_scope2 in final_combined_data.columns and col_scope3 in final_combined_data.columns:
                try:
                    # Convert columns to numeric (in case there are non-numeric values)
                    final_combined_data[col_scope1] = pd.to_numeric(final_combined_data[col_scope1], errors='coerce')
                    final_combined_data[col_scope2] = pd.to_numeric(final_combined_data[col_scope2], errors='coerce')
                    final_combined_data[col_scope3] = pd.to_numeric(final_combined_data[col_scope3], errors='coerce')
    
                    # Extract forecasts for 2030
                    forecast_2030_scope1 = final_combined_data.loc[years == 2030, col_scope1].values[0]
                    forecast_2030_scope2 = final_combined_data.loc[years == 2030, col_scope2].values[0]
                    forecast_2030_scope3 = final_combined_data.loc[years == 2030, col_scope3].values[0]
    
                    # Extract forecasts for 2050
                    forecast_2050_scope1 = final_combined_data.loc[years == 2050, col_scope1].values[0]
                    forecast_2050_scope2 = final_combined_data.loc[years == 2050, col_scope2].values[0]
                    forecast_2050_scope3 = final_combined_data.loc[years == 2050, col_scope3].values[0]
    
                    # Store the forecasts in the dictionary
                    forecast_values[company] = {
                        '2030': [forecast_2030_scope1, forecast_2030_scope2, forecast_2030_scope3],
                        '2050': [forecast_2050_scope1, forecast_2050_scope2, forecast_2050_scope3]
                    }
    
                except (KeyError, IndexError) as e:
                    st.error(f"Error retrieving forecast data for {company}: {e}")
            else:
                st.error(f"Data for {company} is not available in the dataset.")
    

        # If exactly two companies are selected, calculate and display percentage differences
        if len(companies_to_compare) == 2:
            company_1, company_2 = companies_to_compare
    
            st.subheader(f"Percentage Difference between {company_1} and {company_2} (2030 and 2050)")
    
            # Calculate percentage difference for each year and scope
            for year in ['2030', '2050']:
                for i, scope in enumerate(['Scope 1', 'Scope 2', 'Scope 3']):
                    value_1 = forecast_values[company_1][year][i]
                    value_2 = forecast_values[company_2][year][i]
    
                    # Check if the data is available for both companies
                    if 'Data not available' not in [value_1, value_2]:
                        # Calculate percentage difference if the first value is not zero
                        percentage_diff = ((value_2 - value_1) / value_1) * 100 if value_1 != 0 else float('inf')
                        st.write(f"{year} {scope}: {company_1}: {value_1:.3f}, {company_2}: {value_2:.3f}")
                        st.write(f"Percentage Difference: {percentage_diff:.2f}%")
                    else:
                        st.write(f"{year} {scope}: Data not available for comparison.")

    
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
                        # Make predictions for the current scope
                        predictions = predict_model(models[scope_name], fh=30)
                        combined_data = combine_data(historical_data[scope_name], predictions.values.flatten(), f'{comp} {scope}')
                        # Store the predictions and original data for comparison
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
                        
                        # Retrieve the forecast values for the current scope
                        forecast_2030 = predictions.loc['2030'].values.flatten() if '2030' in predictions.index else "2030 data not available"
                        forecast_2050 = predictions.loc['2050'].values.flatten() if '2050' in predictions.index else "2050 data not available"
                        st.write(f"### {scope} Forecast")
                        st.write(f"- **2030 Forecast**: {forecast_2030}")
                        st.write(f"- **2050 Forecast**: {forecast_2050}")

                        #st.write(f"### {scope} Forecast")
                        #st.write(f"- **2030 Forecastwewe**: {final_combined_data.iloc[0:10]}")
                        #st.write(f"- **2050 Forecast**: {forecast_2050}")
                    
                    except Exception as e:
                        st.write(f"Error fetching forecast data for {scope}: {e}")

            else:
                st.warning(f"No data available for {scope} comparison.")

    else:
        # If no company is selected for comparison, show the individual company's scope data
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
                    try:

                         # Set pandas option to display floats without scientific notation
                        pd.set_option('display.float_format', '{:,.3f}'.format)                       
                        
                        forecast_2030 = final_combined_data.loc['2030', f'{scope} Prediction'].values.flatten()[0] if '2030' in predictions.index else "2030 data not available"
                        forecast_2050 = final_combined_data.loc['2050', f'{scope} Prediction'].values.flatten()[0] if '2050' in predictions.index else "2050 data not available"
                        
                        st.write(f"- **2030 Forecast**: {forecast_2030}")
                        st.write(f"- **2050 Forecast**: {forecast_2050}")
                        
                    
                    except Exception as e:
                        st.write(f"Error fetching forecast data for {scope}: {e}")
                    
    # Add User Data Chart if available
    if user_data is not None:
        st.subheader(f'{file_name} (Scope 1, Scope 2, Scope 3)')
        fig_user = px.line(user_data, 
                           x=user_data.index, 
                           y=user_data.columns, 
                           title=f'{file_name} (Scope 1, Scope 2, Scope 3)', 
                           labels={"index": "Year", "value": "Emissions (in metric tons)"})
        st.plotly_chart(fig_user)
        
        
        # Set pandas option to display floats without scientific notation
        pd.set_option('display.float_format', '{:,.3f}'.format)

        # 2030 forecast - for user uploaded data
        forecast_2030_scope1 = str(final_combined_data.iloc[15:16,9:10]).split(" ")[-1]
        forecast_2030_scope2 = str(final_combined_data.iloc[15:16,10:11]).split(" ")[-1]
        forecast_2030_scope3 = str(final_combined_data.iloc[15:16,11:12]).split(" ")[-1]

        st.subheader(f"{file_name} 2030 Forecast")
        st.write(f"Scope 1: {forecast_2030_scope1}")
        st.write(f"Scope 2: {forecast_2030_scope2}")
        st.write(f"Scope 3: {forecast_2030_scope3}")

        # 2050 forecast - for user uploaded data
        forecast_2050_scope1 = str(final_combined_data.iloc[35:36,9:10]).split(" ")[-1]
        forecast_2050_scope2 = str(final_combined_data.iloc[35:36,10:11]).split(" ")[-1]
        forecast_2050_scope3 = str(final_combined_data.iloc[35:36,11:12]).split(" ")[-1]

        st.subheader(f"{file_name} 2050 Forecast")
        st.write(f"Scope 1: {forecast_2050_scope1}")
        st.write(f"Scope 2: {forecast_2050_scope2}")
        st.write(f"Scope 3: {forecast_2050_scope3}")

        #st.dataframe(final_combined_data.iloc[35:36,9:])


# Data Table Tab
with tab3:
# Create the subheader with conditional inclusion
    if uploaded_file:
        subheader_text = f'Carbon Emissions Table including {file_name}'
    else:
        subheader_text = 'Carbon Emissions Table'
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
