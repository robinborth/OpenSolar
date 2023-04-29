import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv


@st.cache_data
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


def get_today() -> datetime.date:
    now = datetime.datetime.now()
    return datetime.date(now.year, now.month, now.day)
