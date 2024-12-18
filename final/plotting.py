"""
Module for Visualizing Kickstarter Data and Business Analysis

This module provides functions for visualizing and analyzing Kickstarter data. It includes 
density plots, analysis of business metrics (e.g., profit, overhead costs, break-even backers), 
and Kickstarter prediction analysis using a pre-fitted regression model.

Key Functions
-------------
1. `plot_density_with_vline`:
    Generates density plots with a vertical line overlay to visualize the distribution of data.
2. `plot_business_variables`:
    Produces density plots for critical business metrics such as break-even backers, overhead costs, and profit.
3. `plot_kickstarter_analysis`:
    Analyzes Kickstarter data and generates visualizations and key metrics, including probabilities, percentiles, 
    and revenue predictions.

Dependencies
------------
- numpy
- matplotlib
- seaborn
- pandas
- scipy.stats.norm
- `compute_business_variables` from the `final.analysis` module

Example Usage
-------------
# Example workflow to generate Kickstarter analysis plots
>>> import pandas as pd
>>> from final.analysis import compute_business_variables
>>> from module_name import plot_density_with_vline, plot_business_variables, plot_kickstarter_analysis
>>> df = pd.read_csv("kickstarter_data.csv")
>>> pledge_value = 100.0
>>> unit_cost = 20.0
>>> overhead = 5000.0
>>> breakEvenPercentageOfGoal = 0.5

# Generate business metric plots
>>> figs = plot_business_variables(df, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal)

# Perform Kickstarter analysis and visualization
>>> model = some_pre_fitted_model
>>> results = plot_kickstarter_analysis(pledge_value, 50000, "Art", df, model)
>>> results[0][0].show()  # Show the Average Pledge plot
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd

from final.analysis import compute_business_variables

def plot_density_with_vline(data, vline_val, title, x_breaks, x_labels, color='#4682B4'):
    """
    Plots a density curve for the given data with a vertical line overlaid.

    Parameters
    ----------
    data : array-like
        Data to plot the density for.
    vline_val : float
        The value where the vertical line will be placed.
    title : str
        Title of the plot.
    x_breaks : list of floats
        Tick positions for the x-axis in log scale.
    x_labels : list of str
        Labels corresponding to the x_breaks.
    color : str, optional
        Color for the density plot and vertical line, default is '#4682B4'.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing the plot.

    Examples
    --------
    >>> data = [10, 100, 1000, 10000]
    >>> fig = plot_density_with_vline(data, 500, "Example Plot", [10, 100, 1000], ["10", "100", "1k"])
    >>> fig.show()
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(np.log10(data), ax=ax, fill=True, color=color, alpha=0.5)
    ax.axvline(np.log10(vline_val), color=color, linestyle='--', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("")
    ax.set_xticks(np.log10(x_breaks))
    ax.set_xticklabels(x_labels)
    fig.tight_layout()
    return fig

def plot_business_variables(userData: pd.DataFrame,
                            pledge_value: float,
                            unit_cost: float,
                            overhead: float,
                            breakEvenPercentageOfGoal: float):
    """
    Generates density plots for break-even backers, overhead costs, and profit.

    Parameters
    ----------
    userData : pd.DataFrame
        A DataFrame containing project data with columns like 'breakQ', 'fCost', and 'profit'.
    pledge_value : float
        The value of a single pledge.
    unit_cost : float
        Cost per unit of the pledge.
    overhead : float
        Fixed overhead cost.
    breakEvenPercentageOfGoal : float
        Break-even percentage relative to the goal.

    Returns
    -------
    tuple
        A tuple of matplotlib.figure.Figure objects:
        - break_fig : Break-even backers plot.
        - overhead_fig : Overhead cost plot.
        - profit_fig : Profit plot.

    Examples
    --------
    >>> userData = pd.DataFrame({'breakQ': [10, 20], 'fCost': [100, 200], 'profit': [50, 60]})
    >>> figs = plot_business_variables(userData, 100, 50, 500, 0.8)
    >>> figs[0].show()  # Show the break-even plot
    """
    _, userBreakEvenQuantity, _, userProfit, userOverhead = compute_business_variables(
        userData, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal
    )

    break_fig = plot_density_with_vline(
        data=userData['breakQ'],
        vline_val=userBreakEvenQuantity,
        title="# of Backers to Break Even",
        x_breaks=[10, 100, 1000, 10000, 100000],
        x_labels=["10", "100", "1k", "10k", "100k"],
        color='plum'
    )

    overhead_fig = plot_density_with_vline(
        data=userData['fCost'],
        vline_val=userOverhead,
        title="Overhead Costs",
        x_breaks=[10, 100, 1000, 10000, 100000],
        x_labels=["$10", "$100", "$1k", "$10k", "$100k"],
        color='#4682B4'
    )

    profit_fig = plot_density_with_vline(
        data=userData['profit'],
        vline_val=userProfit,
        title="Profit",
        x_breaks=[10, 100, 1000, 10000, 100000],
        x_labels=["$10", "$100", "$1k", "$10k", "$100k"],
        color='khaki'
    )

    return break_fig, overhead_fig, profit_fig

def plot_kickstarter_analysis(pledge_value: float,
                              goal_value: float,
                              category_value: str,
                              thisData: pd.DataFrame,
                              model):
    """
    Performs an analysis of Kickstarter data and generates plots and key metrics.

    Parameters
    ----------
    pledge_value : float
        The average pledge value for a project.
    goal_value : float
        The funding goal in USD.
    category_value : str
        The parent category of the project.
    thisData : pd.DataFrame
        The input DataFrame containing Kickstarter project data.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        A pre-fitted regression model for predicting 'percent_of_goal_reached'.

    Returns
    -------
    tuple
        A tuple containing:
        (fig_pledge, fig_goal, fig_percent, fig_revenue,
         pledge_percentile, goal_percentile, prob_goal_is_met, expectedRevenue,
         pred_mean, lower_percent, upper_percent, percent_percentile, percent_interpretation,
         lower_rev, upper_rev, revenue_percentile, revenue_interpretation)
    """
    # Compute percentiles
    pledge_percentile = (thisData['averagePledge'] < pledge_value).mean() * 100
    goal_percentile = (thisData['goal_usd'] < goal_value).mean() * 100

    new_data = pd.DataFrame({
        'averagePledge': [pledge_value],
        'goal_usd': [goal_value],
        'parent_category': [category_value]
    })
    new_data['log_pledge'] = np.log10(new_data['averagePledge'])
    new_data['log_goal'] = np.log10(new_data['goal_usd'])

    pred = model.get_prediction(new_data)
    pred_mean = pred.predicted_mean[0]
    sigma = np.sqrt(model.mse_resid)

    # Probability of meeting the goal
    z_val = (0 - pred_mean)/sigma
    prob_goal_is_met = 1 - norm.cdf(z_val)

    # Expected Revenue
    expectedRevenue = (10**pred_mean)*goal_value

    # Compute 90% CI for percent_of_goal_reached
    z_90 = 1.645
    lower_percent = 10**(pred_mean - z_90*sigma)
    upper_percent = 10**(pred_mean + z_90*sigma)

    # Compute percentile and interpretation for predicted percent_of_goal_reached
    from final.analysis import get_percentile_and_interpretation
    percent_percentile, percent_interpretation = get_percentile_and_interpretation(10**pred_mean, thisData['percent_of_goal_reached'])

    # Compute revenue data
    revenue_data = thisData['percent_of_goal_reached'] * thisData['goal_usd']
    # Compute 90% CI for expected revenue
    log_expected_rev_mean = np.log10(expectedRevenue)
    lower_rev = 10**(log_expected_rev_mean - z_90*sigma)
    upper_rev = 10**(log_expected_rev_mean + z_90*sigma)

    # Compute percentile and interpretation for expected revenue
    revenue_percentile, revenue_interpretation = get_percentile_and_interpretation(expectedRevenue, revenue_data)

    # Plotting figures
    fig_pledge = plot_density_with_vline(
        thisData['averagePledge'], pledge_value, "Average Pledge",
        [10, 100, 1000, 10000],
        ["$10", "$100", "$1k", "$10k"]
    )

    fig_goal = plot_density_with_vline(
        thisData['goal_usd'], goal_value, "Goal",
        [10, 100, 1000, 10000, 100000],
        ["$10", "$100", "$1k", "$10k", "$100k"]
    )

    # Percent of Goal Reached plot
    fig_percent_temp, ax_temp = plt.subplots()
    sns.kdeplot(np.log10(thisData['percent_of_goal_reached'][thisData['percent_of_goal_reached'] > 0]), fill=False, ax=ax_temp)
    emp_lines = ax_temp.get_lines()
    if len(emp_lines) > 0:
        emp_x, emp_y = emp_lines[0].get_data()
    else:
        emp_x, emp_y = [], []
    plt.close(fig_percent_temp)

    xs = np.linspace(pred_mean - 3.5*sigma, pred_mean + 3.5*sigma, 1000)
    ys = norm.pdf(xs, loc=pred_mean, scale=sigma)

    fig_percent, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(np.log10(thisData['percent_of_goal_reached']), fill=True, color='#FFD700', alpha=0.5, ax=ax)
    if len(emp_y) > 0 and max(emp_y)>0 and max(ys)>0:
        scale_factor = max(emp_y)/max(ys)
        ys_scaled = ys*scale_factor
        ax.plot(xs, ys_scaled, color='#FFD700', linewidth=2)
    ax.axvline(pred_mean, color='#FFD700', linestyle='--', linewidth=2)
    ax.set_title("Percent of Goal Reached")
    ax.set_xlabel("% of Goal Reached")
    ax.set_ylabel("")
    ax.set_xticks(np.log10([0.01,0.1,1,10,100,1000]))
    ax.set_xticklabels(["1%", "10%", "100%", "10x", "100x", "1000x"])
    fig_percent.tight_layout()

    # Revenue plot
    fig_revenue_temp, ax_temp = plt.subplots()
    sns.kdeplot(np.log10(revenue_data[revenue_data>0]), fill=False, ax=ax_temp)
    emp_lines2 = ax_temp.get_lines()
    if len(emp_lines2) > 0:
        emp_x2, emp_y2 = emp_lines2[0].get_data()
    else:
        emp_x2, emp_y2 = [], []
    plt.close(fig_revenue_temp)

    xs_money = np.linspace(log_expected_rev_mean - 3.5*sigma, log_expected_rev_mean + 3.5*sigma, 1000)
    ys_money = norm.pdf(xs_money, loc=log_expected_rev_mean, scale=sigma)

    fig_revenue, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(np.log10(revenue_data[revenue_data>0]), fill=True, color='#FF4500', alpha=0.5, ax=ax)
    if len(emp_y2) > 0 and max(emp_y2)>0 and max(ys_money)>0:
        scale_factor2 = max(emp_y2)/max(ys_money)
        ys_money_scaled = ys_money*scale_factor2
        ax.plot(xs_money, ys_money_scaled, color='#FF4500', linewidth=2)
    ax.axvline(log_expected_rev_mean, color='#FF4500', linestyle='--', linewidth=2)
    ax.set_title("Revenue")
    ax.set_xlabel("Revenue ($'s)")
    ax.set_ylabel("")
    ax.set_xticks(np.log10([10,100,1000,10000,100000,1000000]))
    ax.set_xticklabels(["$10", "$100", "$1k", "$10k", "$100k", "$1m"])
    fig_revenue.tight_layout()

    return (fig_pledge, fig_goal, fig_percent, fig_revenue,
            pledge_percentile, goal_percentile, prob_goal_is_met, expectedRevenue,
            pred_mean, lower_percent, upper_percent, percent_percentile, percent_interpretation,
            lower_rev, upper_rev, revenue_percentile, revenue_interpretation)
