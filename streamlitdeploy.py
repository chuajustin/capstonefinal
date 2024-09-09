import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.time_series import load_model, predict_model

# Sidebar information
st.sidebar.header('About this APP')
st.sidebar.markdown("""
This app uses time series models to make predictions and displays the results automatically.
Default data is loaded to showcase the forecast based on historical data.

Select a company or year to view forecasts and historical data.
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
def combine_data(historical, prediction, label, custom_name=None, uploaded_file=False):
    pred_index = pd.date_range(start=historical.index[-1] + pd.DateOffset(1), periods=len(prediction), freq='Y')
    prediction_series = pd.Series(prediction, index=pred_index, name=f'{label} Prediction')

    if uploaded_file:
        original_label = custom_name if custom_name else f'{label} Original'

    else:
        original_label = label

    combined_data = pd.concat([historical, prediction_series], axis=1)
    combined_data.columns = [original_label, f'{label} Prediction']

    return combined_data



# Streamlit App
st.title('''Carbon Emission Predictor
This app is built base on this five tech companies historical emission data for the past 6 years. 

(Meta, Fujitsu, Amazon, Google and Microsoft)''')

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
        combined_data_scope_1 = combine_data(user_data, predictions_scope_1.values.flatten(), model_scope_1, custom_name = original_label, uploaded_file=True)

    if model_scope_2 in models:
        predictions_scope_2 = predict_model(models[model_scope_2], fh=fh)
        combined_data_scope_2 = combine_data(user_data, predictions_scope_2.values.flatten(), model_scope_2, custom_name = original_label, uploaded_file=True)

    if model_scope_3 in models:
        predictions_scope_3 = predict_model(models[model_scope_3], fh=fh)
        combined_data_scope_3 = combine_data(user_data, predictions_scope_3.values.flatten(), model_scope_3, custom_name = original_label, uploaded_file=True)

    # Combine all scopes into a single DataFrame for plotting
    final_combined_data = pd.concat([combined_data_scope_1, combined_data_scope_2, combined_data_scope_3], axis=1)

else:
    # Default historical data if no file is uploaded
    user_data = historical_data[f"{company} Scope 1"]
    
# Tabs for Combined Charts, Individual Scope Charts, and Emission Stats Table
tab1, tab2, tab3 = st.tabs(["Combined Charts", "Individual Scope Charts", "Emission Stats Table (in metrics tons)"])

# Get the relevant model names for the selected company
model_names = [f"{company} Scope 1", f"{company} Scope 2", f"{company} Scope 3"]

combined_data_list = []

for scope in model_names:
    if scope in models:
        predictions = predict_model(models[scope], fh=30)
        combined_data = combine_data(user_data, predictions.values.flatten(), scope)
        combined_data_list.append(combined_data)

# Combine all scopes into a single DataFrame for plotting
final_combined_data = pd.concat(combined_data_list, axis=1)


# Combined Charts Tab
with tab1:
    st.subheader('Carbon Emissions Comparison: Scopes 1, 2, and 3 (Original vs Predictions)')

    # Multi-select widget to choose companies for comparison
    companies_to_compare = st.multiselect('Compare with:', ["Meta", "Fujitsu", "Amazon", "Google", "Microsoft"], key='company_comparison')

    if companies_to_compare:
        combined_data_list = []

        for company in companies_to_compare:
            model_names = [f"{company} Scope 1", f"{company} Scope 2", f"{company} Scope 3"]

            for scope in model_names:
                if scope in models:
                    predictions = predict_model(models[scope], fh=30)
                    # Use the user_data if uploaded, otherwise, use historical data
                    historical_scope_data = user_data if uploaded_file else historical_data[scope]
                    combined_data = combine_data(historical_scope_data, predictions.values.flatten(), scope, custom_name=original_label, uploaded_file=uploaded_file)
                    combined_data_list.append(combined_data)

        if combined_data_list:
            final_combined_data = pd.concat(combined_data_list, axis=1)


            # Chart Type Selection
            chart_type = st.selectbox('Select chart type:', ['Line', 'Bar', 'Scatter'], key='chart_type_selection_combined')

            # Chart rendering logic
            if chart_type == 'Line':
                fig_combined = px.line(final_combined_data,
                                         x=final_combined_data.index,
                                         y=final_combined_data.columns,
                                         title=f'{original_label}: Scopes 1, 2, and 3 (Original vs Predictions)',  # Use original_label in the title
                                         labels={"index": "Year", "value": "Emissions (in metric tons)"})
            elif chart_type == 'Bar':
                fig_combined = px.bar(final_combined_data,
                                        x=final_combined_data.index,
                                        y=final_combined_data.columns,
                                        title=f'{original_label}: Scopes 1, 2, and 3 (Original vs Predictions)',  # Use original_label in the title
                                        labels={"index": "Year", "value": "Emissions (in metric tons)"})
            elif chart_type == 'Scatter':
                fig_combined = px.scatter(final_combined_data,
                                           x=final_combined_data.index,
                                           y=final_combined_data.columns,
                                           title=f'{original_label}: Scopes 1, 2, and 3 (Original vs Predictions)',  # Use original_label in the title
                                           labels={"index": "Year", "value": "Emissions (in metric tons)"})

            st.plotly_chart(fig_combined)
        else:
            # Default data display when no company selected
            fig_combined = px.line(final_combined_data,
                                     x=final_combined_data.index,
                                     y=final_combined_data.columns,
                                     title=f'{original_label}: Scopes 1, 2, and 3 (Original vs Predictions)',  # Use original_label in the title
                                     labels={"index": "Year", "value": "Emissions (in metric tons)"})
            st.plotly_chart(fig_combined)

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
                                         title=f'{original_label} {scope} (Original vs Prediction)',  # Use original_label in the title
                                         labels={"index": "Year", "value": "Emissions (in metric tons)"})
                elif chart_type == 'Bar':
                    fig_scope = px.bar(final_combined_data[[f'{scope} Original', f'{scope} Prediction']],
                                        x=final_combined_data.index,
                                        y=[f'{scope} Original', f'{scope} Prediction'],
                                        title=f'{original_label} {scope} (Original vs Prediction)',  # Use original_label in the title
                                        labels={"index": "Year", "value": "Emissions (in metric tons)"})
                elif chart_type == 'Scatter':
                    fig_scope = px.scatter(final_combined_data[[f'{scope} Original', f'{scope} Prediction']],
                                           x=final_combined_data.index,
                                           y=[f'{scope} Original', f'{scope} Prediction'],
                                           title=f'{original_label} {scope} (Original vs Prediction)',  # Use original_label in the title
                                           labels={"index": "Year", "value": "Emissions (in metric tons)"})

                st.plotly_chart(fig_scope)

# Emission Stats Table Tab
with tab3:
    st.subheader(f'Emission Stats Table for {original_label} (in metric tons)')  # Use original_label in the subheader
    st.dataframe(final_combined_data)

# Download as CSV
csv = final_combined_data.to_csv().encode('utf-8')
st.download_button(label="Download data as CSV", data=csv, file_name=f'{original_label}_emissions_comparison.csv', mime='text/csv')
st.sidebar.markdown(f"Original Label: {original_label}")
st.write(final_combined_data.columns)
