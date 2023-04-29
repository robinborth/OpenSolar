import datetime
import json
from dataclasses import dataclass

import cv2
import numpy as np
import requests


@dataclass
class WeatherInfo:
    longitude: float
    latitude: float
    date: datetime.date
    temperature: float
    sunshine_minutes: int

    @property
    def sunshine_hours(self) -> float:
        return self.sunshine_minutes / 60


def get_weather_info(
    longitude: float,
    latitude: float,
    date: datetime.date,
) -> WeatherInfo:
    """Retur

    Args:
        long (float): _description_
        lat (float): _description_
        timestamp (date): _description_
    """

    temperature = 20.2
    sunshine_minutes = 234

    return WeatherInfo(
        longitude=longitude,
        latitude=latitude,
        date=date,
        temperature=temperature,
        sunshine_minutes=sunshine_minutes,
    )


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
