import datetime
import math
import os

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot
from prophet import Prophet
from pyproj import Transformer

from opensolar.algorithms import panel_energy
from opensolar.segmentation import Roof

NODATA_VALUE = -999
XLLCORNER = 3280500
YLLCORNER = 5237500
NROWS = 866
CELLSIZE = 1000


def WGS84_to_GKZ3(longitude: float, latitude: float) -> tuple:
    from_crs = "EPSG:4326"  # WGS 84
    to_crs = "EPSG:31467"  # Gauss Krüger Zone 3

    # Create transformer object
    transformer = Transformer.from_crs(from_crs, to_crs)

    # Convert latitude and longitude to Gauss Krüger coordinates
    h, r = transformer.transform(latitude, longitude)
    return h, r


def coord_to_grids(long: float, lat: float) -> tuple:
    h, r = WGS84_to_GKZ3(long, lat)
    y, x = math.floor((r - XLLCORNER) / CELLSIZE), NROWS - math.ceil(
        (h - YLLCORNER) / CELLSIZE
    )

    return x, y


def get_val(long, lat, month, year, type):
    x, y = coord_to_grids(long, lat)

    base = os.getcwd()
    fp = f"dataset/radiation_{type}_3y/{type}_{year}{month:02d}.asc"
    full_path = os.path.join(base, fp)
    data = np.loadtxt(full_path, skiprows=28)
    data[data == NODATA_VALUE] = np.nan
    val = data[x, y]

    return val


@st.cache_data
def get_historical_data(long, lat):
    ds, direct, diff = [], [], []
    for year in range(2020, 2023):
        for month in range(1, 13):
            ds.append(f"{year}-{month}")
            direct.append(get_val(long, lat, month, year, "direct"))
            diff.append(get_val(long, lat, month, year, "diff"))
    direct_df = pd.DataFrame.from_dict({"ds": ds, "y": direct})
    diff_df = pd.DataFrame.from_dict({"ds": ds, "y": diff})

    return direct_df, diff_df


def forecast_model(df):
    model = Prophet()
    model.fit(df)
    return model


def get_prediction(model, date):
    months = ((date - datetime.date(2022, 12, 31)).days + 10) // 30
    future = model.make_future_dataframe(periods=months, freq="MS")
    forecast = model.predict(future)
    # model.plot(forecast)
    # pyplot.savefig("fig.jpg")
    # vals = forecast[forecast.ds == date.isoformat()]
    # return vals["yhat"].values[0]
    return forecast


# def get_future_infos(long: float, lat: float, date):
#     """Returns for the given dates all of the predictions"""
#     direct_df, diff_df = get_historical_data(long, lat)
#     direct_model = forecast_model(direct_df)
#     diff_model = forecast_model(diff_df)

# return {
#     "date": date,
#     "direct": get_prediction(direct_model, date),
#     "diffuse": get_prediction(diff_model, date),
# }


@st.cache_data
def get_chart_data(
    roofs: list[Roof],
    longitude: float,
    latitude: float,
    dates: list[datetime.date],
    conversion_efficiency: float = 0.3,
) -> pd.DataFrame:
    """The ammount of kWh the roof can produce.

    For the optimal value there is this map: https://globalsolaratlas.info/map?c=51.330612,10.447998,7&r=DEU

    Args:
        latitude (float): The latitude.
        longitude (float): The longitude.
        date (datetime.date): The date of interest.
        conversion_efficiency: the panel's radiation conversion rate
    """
    total_kWhs: list[float] = []

    direct_df, diff_df = get_historical_data(longitude, latitude)

    direct_model = forecast_model(direct_df)
    diff_model = forecast_model(diff_df)

    direct_radiation = get_prediction(direct_model, dates[-1])
    diffuse_radiation = get_prediction(diff_model, dates[-1])

    df = pd.DataFrame.from_dict(
        {
            "date": dates,
            "diffuse_radiation": diffuse_radiation.tail(len(dates))["yhat"],
            "direct_radiation": direct_radiation.tail(len(dates))["yhat"],
        }
    )

    for _, row in df.iterrows():
        total_kWh = 0.0
        avg_kwh_per_sqm = {
            "diffuse": row["diffuse_radiation"],
            "direct": row["direct_radiation"],
        }
        for roof in roofs:
            total_kWh += panel_energy(
                longitude,
                latitude,
                row["date"],
                avg_kwh_per_sqm,
                roof.total_area,
                roof.tilt_angle,
                0.35,
                conversion_efficiency,
            )
        total_kWhs.append(total_kWh)

    df["kWh"] = total_kWhs
    df["date"] = pd.to_datetime(df["date"])
    df["earning"] = df["kWh"].cumsum() * 0.0769
    total_cost = sum([roof.num_solar_panels * roof.cost_per_panel for roof in roofs])
    df["revenue"] = df["earning"] - total_cost
    # add aditional meta information
    return df
