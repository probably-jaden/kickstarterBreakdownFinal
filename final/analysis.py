import numpy as np
import pandas as pd

def filter_category(data: pd.DataFrame, userCategory: str) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows where the 'parent_category' matches the specified userCategory.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing at least a 'parent_category' column.
    userCategory : str
        The category to filter the DataFrame on.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only rows where 'parent_category' equals userCategory.

    Examples
    --------
    >>> df = pd.DataFrame({'parent_category': ['A', 'B', 'A'], 'value': [1, 2, 3]})
    >>> filter_category(df, 'A')
       parent_category  value
    0               A      1
    2               A      3
    """
    return data[data['parent_category'] == userCategory]


def compute_business_variables(userData: pd.DataFrame,
                               pledge_value: float,
                               unit_cost: float,
                               overhead: float,
                               breakEvenPercentageOfGoal: float):
    """
    Computes business-related financial variables for a set of projects.

    Parameters
    ----------
    userData : pd.DataFrame
        DataFrame containing project-related data. Must include the columns:
        - 'goal_usd' : Target goal in USD.
        - 'averagePledge' : Average pledge value.
    pledge_value : float
        The value of a single pledge in monetary terms.
    unit_cost : float
        The cost per unit to produce or fulfill the pledge.
    overhead : float
        Fixed overhead cost that needs to be covered.
    breakEvenPercentageOfGoal : float
        The percentage of the goal that determines the break-even point.

    Returns
    -------
    tuple
        A tuple containing:
        - pd.DataFrame : Updated DataFrame with computed columns:
            - 'vCost' : Variable cost per project.
            - 'breakQ' : Break-even quantity.
            - 'fCost' : Fixed costs based on break-even analysis.
            - 'profit' : Profit projection.
        - float : User break-even quantity.
        - float : Goal value to cover overhead.
        - float : User profit.
        - float : User overhead cost.

    Examples
    --------
    >>> userData = pd.DataFrame({'goal_usd': [1000], 'averagePledge': [50]})
    >>> compute_business_variables(userData, 100, 30, 500, 0.5)
    (   goal_usd  averagePledge  vCost  breakQ  fCost  profit
     0      1000             50   15.0    10.0   350.0   175.0,
     5.0,
     1000.0,
     175.0,
     500.0)
    """
    unitCostPercentage = unit_cost / pledge_value
    userData = userData.copy()
    userData['vCost'] = userData['averagePledge'] * unitCostPercentage
    goal_value = overhead / (breakEvenPercentageOfGoal * (1 - unitCostPercentage))
    profit = (1 - breakEvenPercentageOfGoal) * goal_value * (1 - unitCostPercentage)

    userData['breakQ'] = (userData['goal_usd'] / userData['averagePledge']) * breakEvenPercentageOfGoal
    userData['fCost'] = userData['breakQ'] * (userData['averagePledge'] - userData['vCost'])
    userData['profit'] = (1 - breakEvenPercentageOfGoal) * userData['goal_usd'] * (1 - unitCostPercentage)

    userBreakEvenQuantity = (goal_value / pledge_value) * breakEvenPercentageOfGoal
    userOverhead = userBreakEvenQuantity * (pledge_value - unit_cost)
    userProfit = profit

    return userData, userBreakEvenQuantity, goal_value, userProfit, userOverhead


def get_percentile_and_interpretation(value, data_series):
    """
    Computes the percentile rank of a given value within a data series and provides an interpretation.

    Parameters
    ----------
    value : float
        The value to be evaluated.
    data_series : pd.Series
        A pandas Series containing numerical data to compare against.

    Returns
    -------
    tuple
        A tuple containing:
        - float : Percentile rank of the value (0-100).
        - str : Interpretation of the percentile:
            - "abnormally low" for values below the 20th percentile.
            - "normal" for values between the 20th and 80th percentiles.
            - "abnormally high" for values above the 80th percentile.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> get_percentile_and_interpretation(3, data)
    (40.0, 'normal')
    """
    pct = (data_series < value).mean() * 100
    if pct < 20:
        interpretation = "abnormally low"
    elif pct < 80:
        interpretation = "normal"
    else:
        interpretation = "abnormally high"
    return pct, interpretation
