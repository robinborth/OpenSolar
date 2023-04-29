"""
The streamlit visualization application for the OpenSolar project.

To run the app use `streamlit run opensolar/app.py` and then click on the link that is
displayed in the terminal.
"""
import datetime

import streamlit as st

from opensolar.chart import create_date_series, create_kWh_altair_plot
from opensolar.search import AddressNotFoundError, get_address_info
from opensolar.segmentation import get_google_maps_image, get_roof_info
from opensolar.utils import load_google_cloud_key

st.set_page_config(page_title="OpenSolar", page_icon="🌤️", layout="centered")

st.markdown(
    """
# Welcome to OpenSolar! 👋

If you like to know how much energiy you could produce when having a solar panal please
enter you `address` in the `text_input` below.
"""
)

api_key = load_google_cloud_key()
example_address = "Römerhofweg 16, 85748 Garching bei München, Germany"

address_input = st.text_input(
    label=f"Search for your address (e.g. {example_address}):",
    value=example_address,
)

if address_input:
    try:
        address = get_address_info(address_input, api_key=api_key)

        # the image of the house
        st.write("### 🏠 Google Maps Image")

        image = get_google_maps_image(
            latitude=address.latitude,
            longitude=address.longitude,
            api_key=api_key,
            zoom=20,
        )
        st.image(image)

        st.write("### Solar Panel Information")
        roofs = get_roof_info(image)
        for roof in roofs:
            roof_col1, roof_col2, roof_col3, roof_col4 = st.columns((4, 1, 1, 1))
            with roof_col1:
                st.slider(
                    label=f"**({roof.orientation})** Number Panels",
                    min_value=0,
                    max_value=len(roof.solar_panels),
                    key=roof.instance_id,
                )

            with roof_col2:
                production = roof.max_square_meter
                st.write(f"{production} kWh")

        # create the time frame
        st.write("### Chart Configuration")

        col1, col2 = st.columns([1, 3])

        with col1:
            options = ["day", "week", "month"]
            step_size = st.selectbox(options=options, label="Select Step Size")

        with col2:
            slider_output = st.slider(min_value=1, max_value=52, label="per week")
            delta = datetime.timedelta(weeks=slider_output)
            dates = create_date_series(delta=delta)

        output_options = st.selectbox(
            options=["kWh", "money"],
            label="Set output metric of interest",
        )

        st.write("### 📊 OpenSolar Chart")
        # display the chart
        chart = create_kWh_altair_plot(
            latitude=address.latitude,
            longitude=address.longitude,
            dates=dates,
        )
        st.altair_chart(chart)

    except AddressNotFoundError:
        st.error(
            """Not able to find the address!
            Please be sure that have the right format and enough information!""",
            icon="🚨",
        )
else:
    st.write("Please select a address!")
