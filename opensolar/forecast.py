import datetime
import math
import os

import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from pyproj import Transformer

NODATA_VALUE = -999
XLLCORNER = 3280500
YLLCORNER = 5237500
NROWS = 866
CELLSIZE = 1000


@st.cache_data
def WGS84_to_GKZ3(longitude: float, latitude: float) -> tuple:
    from_crs = "EPSG:4326"  # WGS 84
    to_crs = "EPSG:31467"  # Gauss Krüger Zone 3

    # Create transformer object
    transformer = Transformer.from_crs(from_crs, to_crs)

    # Convert latitude and longitude to Gauss Krüger coordinates
    h, r = transformer.transform(latitude, longitude)
    return h, r


@st.cache_data
def coord_to_grids(long: float, lat: float) -> tuple:
    h, r = WGS84_to_GKZ3(long, lat)
    y, x = math.floor((r - XLLCORNER) / CELLSIZE), NROWS - math.ceil(
        (h - YLLCORNER) / CELLSIZE
    )

    return x, y


@st.cache_data
def get_val(long, lat, month, year, type):
    x, y = coord_to_grids(long, lat)

    base = os.getcwd()
    fp = f"dataset/radiation_{type}_3y/{type}_{year}{month:02d}.asc"
    full_path = os.path.join(base, fp)
    data = np.loadtxt(full_path, skiprows=28)
    data[data == NODATA_VALUE] = np.nan
    val = data[x, y] if data[x, y] != np.nan else -100

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


@st.cache_data
def forecast_model(df):
    model = Prophet(seasonality_mode="multiplicative")
    model.fit(df)
    return model


@st.cache_data
def get_prediction(model, date):
    days = (date - datetime.date(2022, 12, 31)).days + 10
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    vals = forecast[forecast.ds == date.isoformat()]

    return {
        "actual": vals["yhat"].values[0] / 30,
        "upper": vals["yhat_upper"].values[0] / 30,
        "lower": vals["yhat_lower"].values[0] / 30,
    }


@st.cache_data
def get_future_infos(long, lat, date):
    direct_df, diff_df = get_historical_data(long, lat)
    direct_model = forecast_model(direct_df)
    diff_model = forecast_model(diff_df)
    return {
        "direct": get_prediction(direct_model, date),
        "diffuse": get_prediction(diff_model, date),
    }
