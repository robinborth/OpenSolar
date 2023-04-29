from dataclasses import dataclass

import cv2
import numpy as np
import requests
import streamlit as st


@st.cache_data
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


@dataclass
class SegmentationInstance:
    instance_id: int
    mask: list[tuple[int, int]]

    @property
    def square_meter(self) -> float:
        # TODO we need to look that up, should be calc with mask
        return 2.0


@dataclass
class Roof(SegmentationInstance):
    orientation: str
    solar_panels: list[SegmentationInstance]
    obstacle: list[SegmentationInstance]

    @property
    def used_square_meter(self) -> float:
        total_sum: float = 0.0
        for panel in self.solar_panels:
            total_sum += panel.square_meter
        return total_sum

    @property
    def max_square_meter(self) -> float:
        # TODO we need to calc that exaclty
        total_sum: float = self.used_square_meter
        return total_sum * 1.2

    @property
    def unused_sqare_meter(self) -> float:
        return self.max_square_meter - self.used_square_meter

    @property
    def tilt_angle(self) -> float:
        # TODO make that better and complete
        lookup = {"N": 0, "NNE": 22.5}
        return lookup[self.orientation]

    @property
    def color(self) -> str:
        # TODO make a lookup from self.orientation to color
        return "red"


@st.cache_data
def get_roof_info(image: np.ndarray) -> list[Roof]:
    # TOOD calc here the model
    roof1 = Roof(
        instance_id=1,
        mask=[],
        orientation="NNE",
        solar_panels=[
            SegmentationInstance(instance_id=1, mask=[]),
            SegmentationInstance(instance_id=2, mask=[]),
            SegmentationInstance(instance_id=3, mask=[]),
        ],
        obstacle=[],
    )
    roof2 = Roof(
        instance_id=2,
        mask=[],
        orientation="N",
        solar_panels=[
            SegmentationInstance(instance_id=4, mask=[]),
            SegmentationInstance(instance_id=5, mask=[]),
        ],
        obstacle=[],
    )
    return [roof1, roof2]
