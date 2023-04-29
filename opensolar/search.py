import json
from dataclasses import dataclass

import requests
import streamlit as st


class AddressNotFoundError(Exception):
    pass


@dataclass
class Address:
    formatted: str
    latitude: float
    longitude: float


@st.cache_data
def get_address_info(
    address: str,
    api_key: str,
) -> Address:
    """Gets the address components from a given adress.

    For further usage see: https://developers.google.com/maps/documentation/geocoding/requests-geocoding

    Args:
        address (str): Just the address of interest.
        api_key (str): The api_key that needs to be specified.
    """
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(url).content
    address_infos = json.loads(response)["results"]

    if not address_infos or len(address_infos) > 1:
        raise AddressNotFoundError()

    return Address(
        formatted=address_infos[0]["formatted_address"],
        latitude=address_infos[0]["geometry"]["location"]["lat"],
        longitude=address_infos[0]["geometry"]["location"]["lng"],
    )
