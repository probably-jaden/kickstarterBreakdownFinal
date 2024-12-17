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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the Kickstarter project data to compute additional metrics and filter records.

    The preprocessing steps include:
    - Replacing zero values in 'usd_pledged' and 'backers_count' with 1.
    - Computing derived metrics such as 'goal_usd', 'surplus_usd', 'averagePledge', and 'percent_of_goal_reached'.
    - Adding a binary 'passed' column to indicate whether the goal was reached.
    - Extracting the parent category from the 'category' JSON string.
    - Filtering out projects with fewer than 5 backers.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing Kickstarter project data. Required columns include:
        - 'usd_exchange_rate'
        - 'usd_pledged'
        - 'goal'
        - 'category'
        - 'backers_count'

    Returns
    -------
    pd.DataFrame
        A preprocessed DataFrame with the following additional columns:
        - 'goal_usd' : Goal amount in USD.
        - 'surplus_usd' : Surplus amount (pledged - goal_usd).
        - 'averagePledge' : Average pledge per backer.
        - 'passed' : Binary column indicating whether the project reached its goal.
        - 'percent_of_goal_reached' : Percentage of the goal that was reached.
        - 'parent_category' : Extracted parent category from the 'category' column.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'usd_exchange_rate': [1.0],
    ...     'usd_pledged': [100],
    ...     'goal': [200],
    ...     'category': ['{"parent_name": "Technology"}'],
    ...     'backers_count': [10]
    ... })
    >>> preprocess_data(df)
       usd_exchange_rate  usd_pledged  backers_count  goal_usd  surplus_usd  ...  passed  percent_of_goal_reached parent_category
    0               1.0          100             10     200.0       -100.0  ...       0                    0.50       Technology
    """
    df['usd_pledged'] = df['usd_pledged'].replace(0, 1)
    df['backers_count'] = df['backers_count'].replace(0, 1)
    df['goal_usd'] = df['goal'] * df['usd_exchange_rate']
    df['surplus_usd'] = df['usd_pledged'] - df['goal_usd']
    df['averagePledge'] = df['usd_pledged'] / df['backers_count']
    df['passed'] = np.where(df['surplus_usd'] >= 0, 1, 0)
    df['percent_of_goal_reached'] = df['usd_pledged'] / df['goal_usd']

    def extract_parent_category(cat_str):
        """
        Extracts the parent category name from a JSON-formatted category string.

        Parameters
        ----------
        cat_str : str
            A JSON string containing the 'parent_name' key.

        Returns
        -------
        str or np.nan
            The extracted parent category name, or NaN if extraction fails.
        """
        try:
            parsed = json.loads(cat_str)
            parent_name = parsed.get('parent_name', None)
            if parent_name is None or isinstance(parent_name, bool):
                return np.nan
            return str(parent_name)
        except:
            return np.nan

    df['parent_category'] = df['category'].apply(extract_parent_category)
    # Filter out low-backers projects
    df = df[df['backers_count'] > 5]
    return df
