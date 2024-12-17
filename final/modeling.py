import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

def fit_interaction_model(df: pd.DataFrame):
    """
    Fits an Ordinary Least Squares (OLS) regression model with interaction terms.

    The model predicts the log-transformed 'percent_of_goal_reached' based on:
    - Log-transformed 'averagePledge' (log_pledge),
    - Log-transformed 'goal_usd' (log_goal),
    - Interaction effects between 'log_pledge', 'log_goal', and the categorical variable 'parent_category'.

    The function filters the input DataFrame to exclude rows with non-positive values in the relevant columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the following columns:
        - 'percent_of_goal_reached' : Percentage of the funding goal reached.
        - 'averagePledge' : Average pledge amount per backer.
        - 'goal_usd' : Funding goal in USD.
        - 'parent_category' : Categorical variable for the project category.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        A fitted OLS model with interaction terms.

    Notes
    -----
    - Logarithmic transformations are applied to 'percent_of_goal_reached', 'averagePledge', and 'goal_usd'.
    - Interaction terms are included between 'log_pledge', 'log_goal', and the categorical variable 'parent_category'.

    Examples
    --------
    >>> import pandas as pd
    >>> from statsmodels.formula.api import ols
    >>> df = pd.DataFrame({
    ...     'percent_of_goal_reached': [1.2, 0.8, 1.5],
    ...     'averagePledge': [50, 75, 30],
    ...     'goal_usd': [1000, 1500, 2000],
    ...     'parent_category': ['Technology', 'Music', 'Art']
    ... })
    >>> model = fit_interaction_model(df)
    >>> print(model.summary())
    """
    df = df[(df['percent_of_goal_reached'] > 0) & 
            (df['averagePledge'] > 0) & 
            (df['goal_usd'] > 0)].copy()

    df['log_percent'] = np.log10(df['percent_of_goal_reached'])
    df['log_pledge'] = np.log10(df['averagePledge'])
    df['log_goal'] = np.log10(df['goal_usd'])

    model = ols('log_percent ~ log_pledge * log_goal * C(parent_category)', data=df).fit()
    return model
