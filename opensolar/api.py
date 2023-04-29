import datetime
import json

import cv2
import numpy as np
import requests


def get_kWh_production_dummy(
    longitude: float,
    latitude: float,
    date: datetime.date,
    random_delta: float | None = None,
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

    if random_delta:
        return base_kWh_roof + margin + bias_day_kWh + long_bias + random_delta
    return base_kWh_roof + margin + bias_day_kWh + long_bias


def get_address_info(
    address: str,
    api_key: str,
) -> list[dict]:
    """Gets the address components from a given adress.

    For further usage see: https://developers.google.com/maps/documentation/geocoding/requests-geocoding

    Args:
        address (str): Just the address of interest.
        api_key (str): The api_key that needs to be specified.
    """
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(url).content
    data = json.loads(response)["results"]
    return data


def get_google_maps_image(
    latitude: float,
    longitude: float,
    api_key: str,
    zoom: int = 12,
    maptype: str = "satellite",
    image_size_px: int = 400,
) -> np.ndarray:
    """Given latitude and longitude give the image as numpy array.

    For further usage see: https://developers.google.com/maps/documentation/maps-static/overview

    Args:
        latitude (float): The latitude.
        longitude (float): The longitude.
        api_key (str): The google cloud api key.
        zoom (int, optional): The ammount of zoom, larger value more zoom. Defaults to 12.
        maptype (str, optional): The type of the map. Defaults to "satellite".
        image_size_px (int, optional): The width and height of the image. Defaults to 400.

    Returns:
        np.ndarray: The image as numpy array of shape (image_size_px, image_size_px, 3)
    """
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&maptype={maptype}&size={image_size_px}x{image_size_px}&key={api_key}"
    response = requests.get(url).content
    image = cv2.imdecode(np.frombuffer(response, np.uint8), cv2.IMREAD_UNCHANGED)
    return image
