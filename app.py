"""
The streamlit visualization application for the OpenSolar project.

To run the app use `streamlit run opensolar/app.py` and then click on the link that is
displayed in the terminal.
"""
import datetime

import altair as alt
import streamlit as st

from opensolar.detection import Detector
from opensolar.draw_utils import draw_edge_maps, draw_instance_masks, draw_panels
from opensolar.forecast import get_chart_data
from opensolar.optimizer.solver import place_solar_panels
from opensolar.search import AddressNotFoundError, get_address_info
from opensolar.segmentation import (
    get_cost_metric,
    get_google_maps_image,
    get_production_metric,
    get_roof_info,
    Roof
)
from opensolar.utils import get_next_month_first_dates, load_google_cloud_key

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


@st.cache_resource
def load_model():
    return Detector(
        conf_thres=0.45,
        iou_thres=0.7,
        weight_path="./opensolar/detection/weights/best.pt",
    )


@st.cache_data
def get_infos(image):
    # Process image
    meta_info, pred = detector.detect(image)
    image_segmentation = draw_instance_masks(meta_info, pred)

    roof_panels, edges_maps = place_solar_panels(pred, image)
    image_solar_panels = draw_panels(image.copy(), roof_panels)

    image_segmentation = draw_edge_maps(image_segmentation, edges_maps)

    roofs = []

    for roof in roof_panels:
        if len(roof['panels']) > 0:
            roofs.append(Roof(orientation=roof['orientation'], num_solar_panels=len(roof['panels'])))

    return image_segmentation, image_solar_panels, roofs


detector = load_model()

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

        # Process image
        image_segmentation, image_solar_panels, roofs = get_infos(image)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image)
        with col2:
            st.image(image_segmentation)
        with col3:
            st.image(image_solar_panels)

        # Solar Panel Configuration Block
        st.write("### Solar Panel Configuration")
        if roofs:
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
                _, pcurrent, pdelta = get_production_metric(roofs, ratios)
                st.metric(
                    label="⚡ Production",
                    value=f"{pcurrent:.2f} kWh",
                    delta=f"{pdelta:.2f}%",
                )
            with metric2:
                _, ccurrent, cdelta = get_cost_metric(roofs, ratios)
                st.metric(label="💰 Cost", value=f"{ccurrent:.2f}$", delta=f"-{cdelta:.2f}%")

        # create the time frame
        st.write("### Chart Configuration")
        min_year = 1
        max_year = 2

        start_date = st.date_input("Pick Start Date", value=datetime.date.today())
        num_years = st.slider(
            min_value=min_year,
            max_value=max_year,
            label="Delta In Year",
        )
        dates = get_next_month_first_dates(
            start_date=start_date,
            num_years=max_year,
        )
        metric = st.selectbox(
            options=[
                "kWh",
                "revenue",
                "earning",
                "diffuse_radiation",
                "direct_radiation",
            ],
            label="Select Output Metric",
        )

        full_chart_data = get_chart_data(
            roofs=roofs,
            longitude=address.longitude,
            latitude=address.latitude,
            dates=dates,
        )
        chart_data = full_chart_data[: num_years * 12]
        st.dataframe(chart_data)

        st.write("### 📊 OpenSolar Chart")
        chart = (
            alt.Chart(chart_data)
            .mark_line()
            .encode(
                x="date:T",
                y="kWh:Q",
            )
        )
        st.altair_chart(chart)

        # This is just for centering the metrics
        _, metric1, metric2, metric3 = st.columns((1, 4, 4, 4))
        with metric1:
            total_production = chart_data["kWh"].sum()
            st.metric(
                label="⚡ Total Production",
                value=f"{total_production:.2f} kWh",
                delta=f"{pdelta:.2f}%",
            )
        with metric2:
            total_revenue = chart_data["revenue"].max()
            st.metric(
                label="📈 Total Revenue",
                value=f"{total_revenue:.2f}$",
                delta=f"{total_revenue:.2f}%",
            )
        with metric3:
            tree_per_kwh = 55.3
            trees = total_production // tree_per_kwh
            st.metric(
                label="🌳 Equivalent Trees",
                value=f"{trees}",
                delta=f"{trees}",
            )
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
