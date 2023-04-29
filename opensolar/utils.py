import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


def load_google_cloud_key() -> str:
    """Loads the google cloud key from environment."""
    load_dotenv()
    key = os.environ.get("GOOGLE_CLOUD_KEY")
    if key is None:
        raise KeyError("Please set the GoogleCloudKey in the env.")
    return key


def show_image(image: np.ndarray) -> None:
    """Display the array as an image"""
    plt.imshow(image, cmap="gray")
    plt.show()


def create_date_series(
    start_date: datetime.date | None = None,
    end_delta: datetime.timedelta | None = None,
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

    end_date = start_date + end_delta  # type: ignore

    dates = [start_date]
    delta = datetime.timedelta(days=1)

    current_date = start_date
    while current_date <= end_date:
        current_date += delta
        dates.append(current_date)
    return dates
