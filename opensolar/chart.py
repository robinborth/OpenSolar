import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from opensolar.forecast import get_kWh_production


@st.cache_data
def create_date_series(
    start_date=None,
    delta=None,
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
        kWh = get_kWh_production(
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

    return chart
