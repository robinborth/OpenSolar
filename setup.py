from setuptools import find_packages, setup

setup(
    name="opensolar",
    version="1.0",
    authors=["Yaomengxi Han <yaomengxi.han@tum.de>, Robin Borth <robin.borth@tum.de>"],
    description="The project for the hackathon.",
    packages=find_packages(include=("opensolar")),
)
