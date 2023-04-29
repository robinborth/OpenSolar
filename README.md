# OpenSolar

The project of the TUM.ai Hackathon 2023

## Description

Session-based recommender systems capture short-term and dynamic user preferences in domains like e-commerce and news.
Deep neural networks are effective for modeling complex patterns in session-based recommender systems.
We want to utilize the massive proprietary session-based user-behavior dataset.
However, possible propensity downstream tasks are not predetermined and vary based on the company’s objectives.
Therefore, we want to utilize this project to show that learning a representation for an entire session can benefit downstream tasks and enhance flexibility.

## Requirenments

bash
python>=3.11.2
pytorch>2.0
lightning>2.0


## Instalation

To install the dependencies please follow the instructions below.

### Python Dependencies

Install the python dependencies given in the requirenments.txt file.

```bash
pip install -r requirements.txt
pip install -e .
```

### Pre-commit

In order to use pre-commit follow the official docs.

### Load the SBUBdataset

In order to use the project on your machine you need to download the SBUBdataset.
You then need to download the respective file and put them in the data/sbub folder, such that we have following structure.

batch
data
└── sbub
    └── sample_events.parquet
