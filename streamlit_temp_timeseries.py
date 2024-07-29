import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import json
import os
from io import BytesIO
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request


from schedule_temp_timeseries import load_and_concat_all_sheets_in_centers, clean_data, perform_eda, fit_arima_model, forecast_and_plot

st.set_page_config(page_title="Temp Forecast App", layout="wide")

st.title("Temperature Forecast App")

with open('last_update.txt', 'r') as f:
    last_update = f.read().strip()

st.write(f"Last updated: {last_update}")

def get_credentials():
    # Retrieve the JSON string from the environment variable
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable is not set")

    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError:
        raise ValueError("GOOGLE_CREDENTIALS is not a valid JSON string")

    required_fields = ['client_email', 'token_uri', 'private_key']
    for field in required_fields:
        if field not in creds_dict:
            raise ValueError(f"GOOGLE_CREDENTIALS is missing required field: {field}")

    # Create and return the credentials object
    return Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
    )

@st.cache_data
def load_data():
    base_directory_id = '1uH8e33HJQG4v8BmCZ_EJYmGH-VmWI547'  # Hardcoded
    credentials_path = get_credentials()
    raw_data = load_and_concat_all_sheets_in_centers(base_directory_id, credentials_path)
    return clean_data(raw_data)

with st.spinner("Loading data..."):
    data = load_data()

st.sidebar.header("Model Parameters")
st.sidebar.markdown("""
    Adjust these parameters to fine-tune your forecast:
    - **Trend (P)**: Higher values capture longer-term trends
    - **Seasonality (D)**: Accounts for repeating patterns
    - **Short-term fluctuations (Q)**: Captures rapid changes
""")

p = st.sidebar.slider("Trend (P)", 0, 5, 1, help="Controls the order of the autoregressive term")
d = st.sidebar.slider("Seasonality (D)", 0, 2, 1, help="Controls the degree of differencing")
q = st.sidebar.slider("Short-term fluctuations (Q)", 0, 5, 1, help="Controls the order of the moving average term")

forecast_steps = st.sidebar.slider("Forecast Horizon (weeks)", 1, 52, 25, help="Number of weeks to forecast")

locations = data['Location'].unique()
selected_location = st.selectbox("Select Location", locations)

st.header(f"Temp Forecast for {selected_location}")

# EDA
st.subheader("Exploratory Data Analysis")
fig_ts, adf_results, fig_decomp = perform_eda(data, selected_location)

st.pyplot(fig_ts)

# Display ADF Test Results
st.write("ADF Test Results:")
st.write(f"ADF Test Statistic: {adf_results['ADF Test Statistic']:.4f}")
st.write(f"p-value: {adf_results['p-value']:.4f}")
st.write("Critical Values:")
for key, value in adf_results['Critical Values'].items():
    st.write(f"  {key}: {value:.4f}")

# Display Seasonal Decomposition
st.pyplot(fig_decomp)

st.subheader("ARIMA Model and Forecast")
model = fit_arima_model(data, selected_location, order=(p, d, q))
forecast, dates = forecast_and_plot(model, data, selected_location, steps=forecast_steps)

fig_forecast = plt.figure(figsize=(12, 6))
location_data = data[data['Location'] == selected_location]
plt.plot(location_data['Timestamp'], location_data['Temp01'], marker='o', label=f'Actual {selected_location} Temp')
plt.plot(dates, forecast, linestyle='--', marker='o', color='red', label=f'Forecast {selected_location} Temp')
plt.xlabel('Timestamp')
plt.ylabel('Temp (C)')
plt.title(f'Actual vs Forecast for {selected_location}')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig_forecast)

st.subheader("Forecasted Values")
forecast_df = pd.DataFrame({'Date': dates, 'Forecasted Temp': forecast})
st.dataframe(forecast_df)

csv = forecast_df.to_csv(index=False)
st.download_button(
    label="Download Forecast as CSV",
    data=csv,
    file_name=f"{selected_location}_Temp_forecast.csv",
    mime="text/csv",
)

st.subheader("Model Performance")
st.write(f"AIC: {model.aic:.2f}")
st.write(f"BIC: {model.bic:.2f}")

# Residual analysis
st.subheader("Residual Analysis")
residuals = model.resid
fig_residuals, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(residuals)
ax1.set_title('Residuals over time')
ax2.hist(residuals, bins=20)
ax2.set_title('Histogram of residuals')
st.pyplot(fig_residuals)
