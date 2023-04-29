import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def get_kWh_production_dummy(
    longitude: float,
    latitude: float,
    date: datetime.date,
) -> float:
    """The ammount of kWh the roof can produce.

    For the optimal value there is this map: https://globalsolaratlas.info/map?c=51.330612,10.447998,7&r=DEU

    Args:
        latitude (float): The latitude.
        longitude (float): The longitude.
        date (datetime.date): The date of interest.
    """
    # TODO write the correct function
    avg_roof_size = 200
    avg_kWh_per_sqm = 2.92

    base_kWh_roof = avg_roof_size * avg_kWh_per_sqm

    start_date = datetime.date(date.year, 1, 1)
    day_in_year = (date - start_date).days

    scale = 0.10
    sin_factor = np.sin(
        ((day_in_year - 1) / 365) * 2 * np.pi - np.pi / 2
    ) + np.random.normal(0, 0.1)

    margin = sin_factor * scale * base_kWh_roof

    bias_day = np.abs(date.year - 2020) * 356 + day_in_year
    bias_day_kWh = bias_day * 0.03

    long_bias = np.abs(np.abs(longitude) - 180) * 0.2

    return base_kWh_roof + margin + bias_day_kWh + long_bias


@st.cache_data
def create_date_series(
    start_date: datetime.date | None = None,
    delta: datetime.timedelta | None = None,
) -> list[datetime.date]:
    """Creates a series of dates given a start date.

    Args:
        start_date (datetime.date | None, optional): Start date. Defaults to None.
        end_delta (datetime.timedelta | None, optional): This is just the delta, we get
            the return with start_date + end_delta. Defaults to None.
    """
    if start_date is None:
        now = datetime.datetime.now()
        start_date = datetime.date(now.year, now.month, now.day)

    if delta is None:
        delta = datetime.timedelta(weeks=52)

    end_date = start_date + delta  # type: ignore

    dates = [start_date]
    add_delta = datetime.timedelta(days=1)

    current_date = start_date
    while current_date <= end_date:
        current_date += add_delta
        dates.append(current_date)
    return dates


@st.cache_data
def create_kWh_altair_plot(
    latitude: float,
    longitude: float,
    dates: list[datetime.date],
) -> alt.Chart:
    # TODO remove random delta later
    random_delta = np.random.normal(0, 10)
    kWhs = []
    for date in dates:
        kWh = get_kWh_production_dummy(
            longitude=longitude,
            latitude=latitude,
            date=date,
        )
        kWhs.append(kWh + random_delta)

    chart_data = pd.DataFrame({"date": dates, "kWh": kWhs})
    chart_data["date"] = pd.to_datetime(chart_data["date"])

    chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            x="date:T",
            y="kWh:Q",
        )
    )

    return chart
