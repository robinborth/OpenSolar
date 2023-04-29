"""
The streamlit visualization application for the OpenSolar project.

To run the app use `streamlit run opensolar/app.py` and then click on the link that is
displayed in the terminal.
"""
import streamlit as st

st.set_page_config(
    page_title="OpenSolar",
    page_icon="🌤️",
)


st.write("# Welcome to OpenSolar! 👋")
st.sidebar.header("Main Page")

x = st.slider("x")  # 👈 this is a widget
st.write(x, "squared is", x * x)
