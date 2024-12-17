from final import preprocessing
from final import analysis
from final import modeling 
from final import plotting

import streamlit as st
import pandas as pd
import numpy as np

def main():
    """
    Runs the full Kickstarter analysis pipeline.

    This function demonstrates the end-to-end workflow, including:
    - Data loading and preprocessing.
    - Filtering data for a specific project category.
    - Computing business metrics such as break-even backers, profit, and overhead.
    - Fitting an OLS regression model with interaction terms.
    - Generating plots to visualize business variables and prediction results.

    Workflow Steps
    --------------
    1. **Data Loading**:
        - Loads Kickstarter data using `load_data`.
    2. **Preprocessing**:
        - Cleans the data using `preprocess_data`.
    3. **Filtering**:
        - Filters the data for a specific category (e.g., 'Art') using `filter_category`.
    4. **Business Variable Computation**:
        - Computes key business variables (e.g., break-even quantities) with `compute_business_variables`.
    5. **Modeling**:
        - Fits an OLS regression model to predict the percentage of goal reached using `fit_interaction_model`.
    6. **Visualization**:
        - Generates density plots for business metrics using `plot_business_variables`.
        - Plots Kickstarter analysis results, including predictions and percentiles, using `plot_kickstarter_analysis`.

    Notes
    -----
    - Example inputs are hardcoded, such as the pledge value, unit cost, and overhead costs.
    - Figures are generated but not explicitly displayed or saved in this function.

    Example
    -------
    To execute the script:
    >>> python main_script.py

    Outputs:
    - Density plots for business variables: break-even backers, overhead costs, and profit.
    - Visualizations of pledge values, funding goals, revenue, and percent of goal reached.
    """
    # Example usage:
    df = load_data()
    df_clean = preprocess_data(df)

    # Filter by a specific category (example)
    userCategory = "Art"
    filtered_data = filter_category(df_clean, userCategory)

    # Compute business variables (example values)
    pledge_value = 100.0
    unit_cost = 20.0
    overhead = 5000.0
    breakEvenPercentageOfGoal = 0.5
    userData, userBreakEvenQuantity, goal_value, userProfit, userOverhead = compute_business_variables(
        filtered_data, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal
    )

    # Fit model
    model = fit_interaction_model(filtered_data)

    # Plot business variables
    break_fig, overhead_fig, profit_fig = plot_business_variables(
        filtered_data, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal
    )

    # Plot kickstarter analysis
    category_value = userCategory
    (fig_pledge, fig_goal, fig_percent, fig_revenue,
     pledge_percentile, goal_percentile, prob_goal_is_met, expectedRevenue,
     pred_mean, lower_percent, upper_percent, percent_percentile, percent_interpretation,
     lower_rev, upper_rev, revenue_percentile, revenue_interpretation) = plot_kickstarter_analysis(
        pledge_value, goal_value, category_value, filtered_data, model
    )

if __name__ == "__main__":
    main()
