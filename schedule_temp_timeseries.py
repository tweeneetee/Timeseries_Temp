import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def create_gspread_client(credentials_path):
    return gspread.authorize(credentials_path)

def authenticate_drive(credentials_path):
    return build("drive", "v3", credentials=credentials_path)

def get_sheets_in_folder(drive_service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
    results = (
        drive_service.files()
        .list(q=query, pageSize=1000, fields="files(id, name)")
        .execute()
    )
    return results.get("files", [])


def read_and_concat_sheets(client, file_id, header_row=1):
    spreadsheet = client.open_by_key(file_id)
    all_sheets_data = []
    for sheet in spreadsheet.worksheets():
        sheet_data = pd.DataFrame(sheet.get_all_values())
        # Check if the first row has empty values
        if any(pd.isna(sheet_data.iloc[header_row - 1])):
            sheet_data.columns = sheet_data.iloc[header_row]
        else:
            sheet_data.columns = sheet_data.iloc[header_row - 1]
        sheet_data.reset_index(drop=True, inplace=True)  # Reset index
        final_columns = [
            "Timestamp",
            "Number of Worms (non-counted)",
            "Phosphorous01",
            "Phosphorous02",
            "Nitrogen01",
            "Nitrogen02",
            "Potassium01",
            "Potassium02",
            "Light Intensity",
            "Temp01",
            "Hum01",
            "Heat01",
            "SoilM01",
            "SoilM02",
            "Buzzer",
            "pH Rod 1",
            "pH Rod 2",
        ]
        sheet_data = sheet_data.loc[:, ~sheet_data.columns.duplicated()]
        sheet_data = sheet_data.reindex(columns=final_columns, fill_value=None)
        print(sheet_data.shape)
        # sheet_name = sheet.title
        # print("Sheet Name:", sheet_name)
        all_sheets_data.append(sheet_data)
    # Concatenate all sheet data into a single DataFrame
    return pd.concat(all_sheets_data, ignore_index=True)


def load_and_concat_all_sheets_in_centers(base_directory_id, credentials_path):
    client = create_gspread_client(credentials_path)
    drive_service = authenticate_drive(credentials_path)

    compost = []
    center_folders = (
        drive_service.files()
        .list(
            q=f"'{base_directory_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)",
        )
        .execute()
        .get("files", [])
    )

    for center_folder in center_folders:
        sheet_files = get_sheets_in_folder(drive_service, center_folder["id"])
        for sheet_file in sheet_files:
            sheet_data = read_and_concat_sheets(client, sheet_file["id"])
            sheet_data["Location"] = center_folder["name"].split("_")[1]
            compost.append(sheet_data)

    return pd.concat(compost, ignore_index=True)


def clean_data(data):
    print(data.head())
    data = data.dropna(subset=["Timestamp", "Temp01"])
    data = data[~data["Timestamp"].str.contains("Unit|Timestamp")]
    data = data[data["Timestamp"].notna()]
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["Temp01"] = pd.to_numeric(data["Temp01"], errors="coerce")
    data = data[data["Temp01"] > 0]
    data = data[data["Timestamp"].dt.year >= 2024]
    return data


def perform_eda(data, location):
    location_data = data[data["Location"] == location]

    # Time Series Plot
    fig_ts, ax_ts = plt.subplots(figsize=(12, 6))
    ax_ts.plot(location_data["Timestamp"], location_data["Temp01"], marker="o")
    ax_ts.set_xlabel("Timestamp")
    ax_ts.set_ylabel("Temp")
    ax_ts.set_title(f"Time Series Plot for {location}")
    ax_ts.grid(True)

    # ADF Test
    result = adfuller(location_data["Temp01"].dropna())
    adf_results = {
        "ADF Test Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4],
    }

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        location_data.set_index("Timestamp")["Temp01"], model="additive", period=4
    )
    fig_decomp = decomposition.plot()
    fig_decomp.suptitle(f"Seasonal Decomposition for {location}")

    return fig_ts, adf_results, fig_decomp


def fit_arima_model(data, location, order):
    location_data = data[data["Location"] == location]["Temp01"]
    model = ARIMA(location_data, order=order)
    return model.fit()


def forecast_and_plot(model, data, location, steps=25):
    forecast = model.forecast(steps=steps)

    location_data = data[data["Location"] == location]
    last_date = location_data["Timestamp"].max()
    next_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=steps, freq="W"
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        location_data["Timestamp"],
        location_data["Temp01"],
        marker="o",
        label=f"Actual {location} Temp",
    )
    plt.plot(
        next_dates,
        forecast,
        linestyle="--",
        marker="o",
        color="red",
        label=f"Forecast {location} Temp",
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Temp (C) ")
    plt.title(f"Actual vs Forecast for {location}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return forecast, next_dates

def get_credentials():
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable is not set")

    # Parse the JSON string
    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError:
        raise ValueError("GOOGLE_CREDENTIALS is not a valid JSON string")

    # Check for required fields
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


def main():
    base_directory_id = "1uH8e33HJQG4v8BmCZ_EJYmGH-VmWI547"
    credentials_path = get_credentials()

    raw_data = load_and_concat_all_sheets_in_centers(
        base_directory_id, credentials_path
    )
    cleaned_data = clean_data(raw_data)

    locations = cleaned_data["Location"].unique()

    for location in locations:
        perform_eda(cleaned_data, location)

        model = fit_arima_model(cleaned_data, location, order=(0, 1, 0))

        forecast, dates = forecast_and_plot(model, cleaned_data, location)

        print(f"Forecasted Temp values for {location}:")
        for date, value in zip(dates, forecast):
            print(f"{date.strftime('%Y-%m-%d')}: {value:.2f}")


if __name__ == "__main__":
    main()
