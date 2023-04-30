"""
The streamlit visualization application for the OpenSolar project.

To run the app use `streamlit run opensolar/app.py` and then click on the link that is
displayed in the terminal.
"""
import datetime

import streamlit as st

from opensolar.chart import create_date_series, create_kWh_altair_plot
from opensolar.search import AddressNotFoundError, get_address_info
from opensolar.segmentation import (
    get_cost_metric,
    get_google_maps_image,
    get_production_metric,
    get_roof_info,
)
from opensolar.utils import load_google_cloud_key

st.set_page_config(page_title="OpenSolar", page_icon="🌤️")

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
            image_size_px=400,
        )
        image_segmentation = image.copy()
        image_solar_panels = image.copy()

        # The meta data of the image
        roofs = get_roof_info(image)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image)
        with col2:
            st.image(image_segmentation)
        with col3:
            st.image(image_solar_panels)

        # Solar Panel Configuration Block
        st.write("### Solar Panel Configuration")
        roof_cols = st.columns(len(roofs))
        ratios: list = []
        for index, (roof, col) in enumerate(zip(roofs, roof_cols, strict=True)):
            with col:
                sl1 = st.slider(
                    label=f"**({roof.orientation})** Number Panels",
                    value=roof.num_solar_panels,
                    min_value=0,
                    max_value=roof.num_solar_panels,
                    key=index,
                )
                ratios.append(sl1 / roof.num_solar_panels)

        # This is just for centering the metrics
        _, metric1, metric2, _ = st.columns((2, 3, 3, 1))
        with metric1:
            # TODO make this correct
            _, pcurrent, pdelta = get_production_metric(roofs, ratios)
            st.metric(
                label="⚡ Production",
                value=f"{pcurrent:.2f} kWh",
                delta=f"{pdelta:.2f}%",
            )
        with metric2:
            # TODO make this correct current selected one
            _, ccurrent, cdelta = get_cost_metric(roofs, ratios)
            st.metric(label="💰 Cost", value=f"{ccurrent:.2f}$", delta=f"-{cdelta:.2f}%")

        # create the time frame
        st.write("### Chart Configuration")

        today = datetime.date.today()
        start_date = st.date_input("Pick Start Date", value=today)

        metric = st.selectbox(
            options=["kWh", "revenue"],
            label="Select Output Metric",
        )

        slider_output = st.slider(
            min_value=1,
            max_value=25,
            label="Delta In Year",
        )

        st.write("### 📊 OpenSolar Chart")

        # display the chart
        delta = datetime.timedelta(weeks=slider_output)
        dates = create_date_series(delta=delta)
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
    st.write(
        """Please select a address!
            Please be sure that have the right format and enough information!""",
        icon="🚨",
    )
