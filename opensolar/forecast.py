import datetime
import math
import os

import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from pyproj import Transformer

from opensolar.algorithms import panel_energy

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
    model = Prophet(seasonality_mode="multiplicative")
    model.fit(df)
    return model


def get_prediction(model, date):
    days = (date - datetime.date(2022, 12, 31)).days + 30
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    vals = forecast[forecast.ds == date.isoformat()]
    return vals["yhat"].values[0]


@st.cache_data
def get_future_infos(long: float, lat: float, date):
    """Returns for the given dates all of the predictions"""
    direct_df, diff_df = get_historical_data(long, lat)
    direct_model = forecast_model(direct_df)
    diff_model = forecast_model(diff_df)

    return {
        "date": date,
        "direct": get_prediction(direct_model, date),
        "diffuse": get_prediction(diff_model, date),
    }

    # out_infos = []
    # for date in dates:
    #     out_infos.append(
    #         {
    #             "date": date,
    #             "direct": get_prediction(direct_model, date),
    #             "diffuse": get_prediction(diff_model, date),
    #         }
    #     )
    # return out_infos


@st.cache_data
def get_kWh_production(
    longitude: float,
    latitude: float,
    date: datetime.date,
    conversion_efficiency: float = 0.3,
) -> float:
    """The ammount of kWh the roof can produce.

    For the optimal value there is this map: https://globalsolaratlas.info/map?c=51.330612,10.447998,7&r=DEU

    Args:
        latitude (float): The latitude.
        longitude (float): The longitude.
        date (datetime.date): The date of interest.
        conversion_efficiency: the panel's radiation conversion rate
    """
    # roofs = get_roof_info(longitude, latitude)
    roofs = [{"panel_area": 40, "direction": 120}, {"panel_area": 30, "direction": 300}]
    avg_kwh_per_sqm = get_future_infos(longitude, latitude, date)

    base_kWh_roof = 0.0
    for roof in roofs:
        base_kWh_roof += panel_energy(
            longitude,
            latitude,
            date,
            avg_kwh_per_sqm,
            roof["panel_area"],
            roof["direction"],
            0.35,
            conversion_efficiency,
        )
    return base_kWh_roof
