# kickstarterBreakdown

Welcome to the **Kickstarter Analysis** project! This repository contains a Python package and a Streamlit web application for analyzing Kickstarter projects, understanding business variables, and predicting the probability of success for crowdfunding campaigns.

Best way to get familiar with the repo is to play around on the streamlit app, which can be visited here: https://kickstarter.streamlit.app/

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [Using the Python Package](#using-the-python-package)
- [Package Modules](#package-modules)
- [Example Use Cases](#example-use-cases)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This project allows users to:
1. Load and preprocess Kickstarter datasets.
2. Analyze business metrics such as break-even points, profit, and overhead costs.
3. Visualize Kickstarter project distributions (e.g., pledge amounts, backer counts).
4. Predict the likelihood of meeting funding goals using interaction models.
5. Explore Kickstarter data interactively using a **Streamlit web app**.

The tool is ideal for creators, analysts, and data enthusiasts to make informed decisions about crowdfunding campaigns.

---

## Features

- **Load & Clean Data**: Preprocess raw Kickstarter datasets for analysis.
- **Business Metrics**: Compute overhead costs, break-even backers, and profits.
- **Visualizations**: Generate KDE plots, revenue distributions, and 3D interactive visualizations.
- **Prediction**: Use regression models to estimate funding success probability.
- **Streamlit Dashboard**: Explore data interactively via a clean and intuitive UI.

---

## Repository Structure

```plaintext
kickstarter_analysis/
│
├── final/              # Python package directory
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py       # Data loading and preprocessing functions
│   ├── analysis.py            # Business variable computations
│   ├── modeling.py            # Regression models and predictions
│   ├── plotting.py            # Visualization functions
│
├── app/                       # Streamlit app directory
│   └── app.py                 # Main Streamlit script
├── build
│   ├── doctrees
│   └── html
│       ├── _sources
│       │   ├── final.rst.txt
│       │   ├── index.rst.txt
│       │   └── modules.rst.txtgi
├── kickstarterData.csv
├── requirements.txt           # List of Python dependencies
├── setup.py                   # Package installation configuration
├── README.md                  # Documentation
└── LICENSE                    # License file
```


## Installation
Clone the Repository

```plaintext
git clone https://github.com/yourusername/kickstarter_analysis.git
cd kickstarter_analysis
```

### Install Dependencies
Use pip to install the required packages:

```plaintext
pip install -r requirements.txt
```

### Install the Package
Install the package locally in editable mode:

```plaintext
pip install -e .
```

## Usage
## Running the Streamlit App
The Streamlit app provides an interactive dashboard to explore and analyze Kickstarter data.

Navigate to the app/ directory.

Run the Streamlit app:

```plaintext
streamlit run app.py
```

Alternatively view the app at https://kickstarter.streamlit.app/ 

## Using the Python Package
You can also use the Python package programmatically in your own scripts.

Example Script

```plaintext
from your_package.preprocessing import load_data, preprocess_data
from your_package.analysis import filter_category, compute_business_variables
from your_package.plotting import plot_business_variables

# Load and preprocess data
df = load_data()
df_clean = preprocess_data(df)

# Filter data by a specific category
category = "Art"
filtered_data = filter_category(df_clean, category)

# Compute business variables
pledge_value = 50.0
unit_cost = 20.0
overhead = 5000.0
break_even = 0.8

userData, breakEvenQ, goal_value, profit, overhead_cost = compute_business_variables(
    filtered_data, pledge_value, unit_cost, overhead, break_even
)

# Visualize results
break_fig, overhead_fig, profit_fig = plot_business_variables(
    userData, pledge_value, unit_cost, overhead, break_even
)

# Show plots
break_fig.show()
overhead_fig.show()
profit_fig.show()
```

## Package Modules
1. preprocessing.py
load_data: Loads the Kickstarter dataset.
preprocess_data: Cleans and preprocesses the data for analysis.
2. analysis.py
filter_category: Filters the dataset by a user-specified parent category.
compute_business_variables: Computes key business metrics, such as break-even backers, profit, and overhead.
get_percentile_and_interpretation: Provides percentile rankings and interpretations for metrics.
3. modeling.py
fit_interaction_model: Fits an interaction regression model to predict the percentage of goal reached.
4. plotting.py
plot_density_with_vline: Generates KDE plots with vertical lines for key values.
plot_business_variables: Visualizes break-even backers, profit, and overhead costs.
plot_kickstarter_analysis: Produces combined visualizations and predictions.


## Example Use Cases

Estimate the Funding Goal: Given average pledge values and unit costs, compute a goal to maximize success.
Break-even Analysis: Determine the minimum number of backers required to break even for a given campaign.
Probability of Success: Predict the likelihood of meeting the funding goal using a statistical model.
Visualize Trends: Analyze and visualize pledge amounts, backer distributions, and revenue data.


## Dependencies
Python 3.8+
pandas
numpy
streamlit
matplotlib
seaborn
scipy
statsmodels
plotly


## License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Thank you for exploring Kickstarter Analysis! If you encounter issues or have feedback, please open an issue or submit a pull request.

Contact
For questions or support, contact:

Your Name
Email: oldofme3@gmail.com
GitHub: https://github.com/probably-jaden
