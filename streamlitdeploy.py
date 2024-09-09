import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.time_series import load_model, predict_model

# Sidebar information
st.sidebar.header('About this APP')
st.sidebar.markdown("""
This app uses time series models to make predictions and displays the results automatically.
Default data is loaded to showcase the forecast based on historical data.

Select a company to view forecasts and historical data.
""")

# Define model paths and historical data paths
model_paths = {
    # Define your model paths here
}

historical_data_paths = {
    # Define your historical data paths here
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
def combine_data(historical, prediction, label, custom_name=None, uploaded_file=False):
    pred_index = pd.date_range(start=historical.index[-1] + pd.DateOffset(1), periods=len(prediction), freq='Y')
    prediction_series = pd.Series(prediction, index=pred_index, name=f'Prediction {label}')
    
    if uploaded_file:
        # Use custom name for the uploaded file
        original_label = custom_name if custom_name else f'{label} Original'
        combined_data = pd.concat([historical, prediction_series], axis=0)
        combined_data.columns = [original_label, f'{label} Prediction']
    else:
        # Handle the case with default data
        combined_data = pd.concat([historical, prediction_series], axis=0)
        combined_data.columns = [f'{label} Original', f'{label} Prediction']
    
    return combined_data

# Streamlit App Title
st.title('''Carbon Emission Predictor
This app is built based on five tech companies' historical emission data for the past 6 years. 

(Meta, Fujitsu, Amazon, Google, and Microsoft)''')

# User input for company and year
company = st.sidebar.selectbox('Select a company:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], index=0)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file for comparison", type=["csv"])

if uploaded_file is not None:
    # Load user-uploaded data
    user_data = pd.read_csv(uploaded_file, index_col='Year', parse_dates=True)

    # Extract the filename (without extension) for the custom label
    original_label = uploaded_file.name.split('.')[0]  # Get filename without extension
    
    # Assume that the CSV contains the correct format, process the data and generate new predictions
    model_scope_1 = f"{company} Scope 1"
    model_scope_2 = f"{company} Scope 2"
    model_scope_3 = f"{company} Scope 3"

    # Define forecasting horizon (e.g., predict the next 30 years)
    fh = 30

    if model_scope_1 in models:
        predictions_scope_1 = predict_model(models[model_scope_1], fh=fh)
        combined_data_scope_1 = combine_data(user_data, predictions_scope_1.values.flatten(), model_scope_1, custom_name=original_label, uploaded_file=True)

    if model_scope_2 in models:
        predictions_scope_2 = predict_model(models[model_scope_2], fh=fh)
        combined_data_scope_2 = combine_data(user_data, predictions_scope_2.values.flatten(), model_scope_2, custom_name=original_label, uploaded_file=True)

    if model_scope_3 in models:
        predictions_scope_3 = predict_model(models[model_scope_3], fh=fh)
        combined_data_scope_3 = combine_data(user_data, predictions_scope_3.values.flatten(), model_scope_3, custom_name=original_label, uploaded_file=True)

    # Combine all scopes into a single DataFrame for plotting
    final_combined_data = pd.concat([combined_data_scope_1, combined_data_scope_2, combined_data_scope_3], axis=1)

else:
    # Default historical data if no file is uploaded
    user_data = historical_data[f"{company} Scope 1"]

# Tabs for Combined Charts, Individual Scope Charts, and Emission Stats Table
tab1, tab2, tab3 = st.tabs(["Combined Charts", "Individual Scope Charts", "Emission Stats Table (in metrics tons)"])

# Combined Charts Tab
with tab1:
    st.subheader(f'{company}: Scopes 1, 2, and 3 (Original vs Predictions)')

    if uploaded_file:
        # Display a dynamic title using the uploaded file name
        fig_combined = px.line(final_combined_data, 
                               x=final_combined_data.index, 
                               y=final_combined_data.columns, 
                               title=f'{original_label}: Scopes 1, 2, and 3 (Original vs Predictions)', 
                               labels={"index": "Year", "value": "Emissions (in metric tons)"})
    else:
        fig_combined = px.line(final_combined_data, 
                               x=final_combined_data.index, 
                               y=final_combined_data.columns, 
                               title=f'{company}: Scopes 1, 2, and 3 (Original vs Predictions)', 
                               labels={"index": "Year", "value": "Emissions (in metric tons)"})

    st.plotly_chart(fig_combined)


# Individual Scope Charts Tab
# Individual Scope Charts Tab
with tab2:
    companies_to_compare = st.multiselect('Compare with:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], key='company_comparison_indiv')

    if companies_to_compare:
        st.subheader('Comparison of Selected Companies')

        # Chart Type Selection
        chart_type = st.selectbox('Select chart type:', ['Line', 'Bar', 'Scatter'], key='chart_type_selection_indiv')

        for scope in ['Scope 1', 'Scope 2', 'Scope 3']:
            scope_data = pd.DataFrame()

            for comp in companies_to_compare:
                comp_model_name = f"{comp} {scope}"

                if comp_model_name in models:
                    predictions = predict_model(models[comp_model_name], fh=30)
                    historical_scope_data = user_data if uploaded_file else historical_data[comp_model_name]
                    combined_data = combine_data(historical_scope_data, predictions.values.flatten(), comp_model_name, custom_name=original_label, uploaded_file=uploaded_file)
                    scope_data = pd.concat([scope_data, combined_data], axis=1)

            if not scope_data.empty:
                st.subheader(f'{scope} Original', f'{scope} Prediction')

                if chart_type == 'Line':
                    st.line_chart(scope_data)
                elif chart_type == 'Bar':
                    st.bar_chart(scope_data)
                elif chart_type == 'Scatter':
                    fig = px.scatter(scope_data)
                    st.plotly_chart(fig)
    else:
        for scope in model_names:
            st.subheader(f'{company} {scope} (Original vs Prediction)')

            if f'{scope} Original' in final_combined_data.columns and f'{scope} Prediction' in final_combined_data.columns:
                # Chart Type Selection
                chart_type = st.selectbox(f'Select chart type for {scope}:', ['Line', 'Bar', 'Scatter'], key=f'chart_type_selection_{scope}')

                if chart_type == 'Line':
                    fig_scope = px.line(final_combined_data[[f'{scope} Original', f'{scope} Prediction']],
                                        x=final_combined_data.index,
                                        y=[f'{scope} Original', f'{scope} Prediction'],
                                        title=f'{company} {scope} (Original vs Prediction)',
                                        labels={"index": "Year", "value": "Emissions (in metric tons)"})
                elif chart_type == 'Bar':
                    fig_scope = px.bar(final_combined_data[[f'{scope} Original', f'{scope} Prediction']],
                                       x=final_combined_data.index,
                                       y=[f'{scope} Original', f'{scope} Prediction'],
                                       title=f'{company} {scope} (Original vs Prediction)',
                                       labels={"index": "Year", "value": "Emissions (in metric tons)"})
                elif chart_type == 'Scatter':
                    fig_scope = px.scatter(final_combined_data[[f'{scope} Original', f'{scope} Prediction']],
                                           x=final_combined_data.index,
                                           y=[f'{scope} Original', f'{scope} Prediction'],
                                           title=f'{company} {scope} (Original vs Prediction)',
                                           labels={"index": "Year", "value": "Emissions (in metric tons)"})

                st.plotly_chart(fig_scope)


# Emission Stats Table Tab
with tab3:
    st.subheader(f'Emission Stats Table for {company} (in metric tons)')
    st.dataframe(final_combined_data)

# Download as CSV
csv = final_combined_data.to_csv().encode('utf-8')
st.download_button(label="Download data as CSV", data=csv, file_name=f'{company}_emissions_comparison.csv', mime='text/csv')
