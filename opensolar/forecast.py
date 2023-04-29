import pandas as pd
from prophet import Prophet
import numpy as np
import math
from pyproj import Transformer

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


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
    if type == "duration":
        fp = f"dataset/duration/{months[month-1]}-{year}.asc"
        data = np.loadtxt(fp, skiprows=6)
    elif type == "radiation":
        fp = f"dataset/radiation/grids_germany_monthly_radiation_global_{year}{month:02d}.asc"
        data = np.loadtxt(fp, skiprows=28)
    data[data == NODATA_VALUE] = np.nan
    val = data[x, y] if data[x, y] != np.nan else 200

    return val


def get_historical_data(long, lat):
    ds, duration, radiation = [], [], []
    for year in range(2013, 2023):
        for month in range(1, 13):
            ds.append(f"{year}-{month}")
            duration.append(get_val(long, lat, month, year, "duration"))
            radiation.append(get_val(long, lat, month, year, "radiation"))
    duration_df = pd.DataFrame.from_dict({"ds": ds, "y": duration})
    radiation_df = pd.DataFrame.from_dict({"ds": ds, "y": radiation})

    return duration_df, radiation_df


def forecast_model(df):
    model = Prophet(seasonality_mode="multiplicative")
    model.fit(df)
    return model


def get_prediction(model, date):
    days = 365 * (date.year - 2023) + 30 * (date.month) + date.day + 20
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    vals = forecast[forecast.ds == date.isoformat()]

    return {
        "actual": vals["yhat"].values[0],
        "upper": vals["yhat_upper"].values[0],
        "lower": vals["yhat_lower"].values[0],
    }


def get_future_infos(long, lat, date):
    duration_df, radiation_df = get_historical_data(long, lat)
    duration_model = forecast_model(duration_df)
    radiation_model = forecast_model(radiation_df)
    return {
        "sun_duration": get_prediction(duration_model, date),
        "global_radiation": get_prediction(radiation_model, date),
    }
