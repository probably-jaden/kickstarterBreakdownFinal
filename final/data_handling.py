import pandas as pd
import numpy as np
import json

def load_data():
    """
    Loads and processes Kickstarter project data from a CSV file.

    The function performs the following operations:
    - Reads data from 'kickstarterData.csv' into a DataFrame.
    - Converts 'usd_exchange_rate', 'usd_pledged', and 'goal' columns to numeric, coercing invalid values to NaN.
    - Selects relevant columns and removes duplicate entries based on the 'slug' column.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame containing the following columns:
        - 'usd_exchange_rate' : The exchange rate to USD.
        - 'usd_pledged' : The amount pledged in USD.
        - 'category' : The project category.
        - 'urls' : URL information about the project.
        - 'creator' : Information about the project's creator.
        - 'goal' : The funding goal for the project.
        - 'deadline' : The project deadline.
        - 'created_at' : The project creation date.
        - 'slug' : A unique project identifier.
        - 'name' : The project name.
        - 'backers_count' : Number of backers for the project.

    Examples
    --------
    >>> df = load_data()
    >>> df.head()
       usd_exchange_rate  usd_pledged     category                    urls  ...  name  backers_count
    0               1.0         500.0  technology   {'web': 'example.com'}  ...   App           10
    1               1.2         300.0       music  {'web': 'example.org'}   ...  Song            5
    """
    df_combined = pd.read_csv("kickstarterData.csv")
    df_combined['usd_exchange_rate'] = pd.to_numeric(df_combined['usd_exchange_rate'], errors='coerce')
    df_combined['usd_pledged'] = pd.to_numeric(df_combined['usd_pledged'], errors='coerce')
    df_combined['goal'] = pd.to_numeric(df_combined['goal'], errors='coerce')

    df_combined = df_combined[['usd_exchange_rate', 'usd_pledged', 'category', 'urls', 'creator', 'goal', 
                               'deadline', 'created_at', 'slug', 'name', 'backers_count']].drop_duplicates(subset=['slug'])
    return df_combined
