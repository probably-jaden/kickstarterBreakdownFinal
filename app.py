import streamlit as st
import pandas as pd
import numpy as np
import json
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Kickstarter Analysis", layout="wide")

@st.cache_data
def load_data():
    df_combined = pd.read_csv("kickstarterData.csv")
    df_combined['usd_exchange_rate'] = pd.to_numeric(df_combined['usd_exchange_rate'], errors='coerce')
    df_combined['usd_pledged'] = pd.to_numeric(df_combined['usd_pledged'], errors='coerce')
    df_combined['goal'] = pd.to_numeric(df_combined['goal'], errors='coerce')

    df_combined = df_combined[['usd_exchange_rate', 'usd_pledged', 'category', 'urls', 'creator', 'goal', 
                               'deadline', 'created_at', 'slug', 'name', 'backers_count']].drop_duplicates(subset=['slug'])
    return df_combined

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['usd_pledged'] = df['usd_pledged'].replace(0, 1)
    df['backers_count'] = df['backers_count'].replace(0, 1)
    df['goal_usd'] = df['goal'] * df['usd_exchange_rate']
    df['surplus_usd'] = df['usd_pledged'] - df['goal_usd']
    df['averagePledge'] = df['usd_pledged'] / df['backers_count']
    df['passed'] = np.where(df['surplus_usd'] >= 0, 1, 0)
    df['percent_of_goal_reached'] = df['usd_pledged'] / df['goal_usd']

    def extract_parent_category(cat_str):
        try:
            parsed = json.loads(cat_str)
            parent_name = parsed.get('parent_name', None)
            if parent_name is None or isinstance(parent_name, bool):
                return np.nan
            return str(parent_name)
        except:
            return np.nan

    df['parent_category'] = df['category'].apply(extract_parent_category)
    df = df[df['backers_count'] > 5]  # Filter out low-backers projects
    return df

def filter_category(data: pd.DataFrame, userCategory: str) -> pd.DataFrame:
    return data[data['parent_category'] == userCategory]

def compute_business_variables(userData: pd.DataFrame,
                               pledge_value: float,
                               unit_cost: float,
                               overhead: float,
                               breakEvenPercentageOfGoal: float):
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

def plot_density_with_vline(data, vline_val, title, x_breaks, x_labels, color='#4682B4'):
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

def get_percentile_and_interpretation(value, data_series):
    pct = (data_series < value).mean()*100
    if pct < 20:
        interpretation = "abnormally low"
    elif pct < 80:
        interpretation = "normal"
    else:
        interpretation = "abnormally high"
    return pct, interpretation

def plot_business_variables(userData: pd.DataFrame,
                            pledge_value: float,
                            unit_cost: float,
                            overhead: float,
                            breakEvenPercentageOfGoal: float):
    # Compute values for plotting lines
    _, userBreakEvenQuantity, _, userProfit, userOverhead = compute_business_variables(
        userData, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal
    )

    # Break Even plot
    break_fig = plot_density_with_vline(
        data=userData['breakQ'],
        vline_val=userBreakEvenQuantity,
        title="# of Backers to Break Even",
        x_breaks=[10,100,1000,10000,100000],
        x_labels=["10", "100", "1k", "10k", "100k"],
        color='plum'
    )

    # Overhead plot
    overhead_fig = plot_density_with_vline(
        data=userData['fCost'],
        vline_val=userOverhead,
        title="Overhead Costs",
        x_breaks=[10,100,1000,10000,100000],
        x_labels=["$10", "$100", "$1k", "$10k", "$100k"],
        color='#4682B4'
    )

    # Profit plot
    profit_fig = plot_density_with_vline(
        data=userData['profit'],
        vline_val=userProfit,
        title="Profit",
        x_breaks=[10,100,1000,10000,100000],
        x_labels=["$10", "$100", "$1k", "$10k", "$100k"],
        color='khaki'
    )

    return break_fig, overhead_fig, profit_fig

def fit_interaction_model(df: pd.DataFrame):
    df = df[(df['percent_of_goal_reached']>0) & (df['averagePledge']>0) & (df['goal_usd']>0)].copy()
    df['log_percent'] = np.log10(df['percent_of_goal_reached'])
    df['log_pledge'] = np.log10(df['averagePledge'])
    df['log_goal'] = np.log10(df['goal_usd'])
    model = ols('log_percent ~ log_pledge * log_goal * C(parent_category)', data=df).fit()
    return model

def plot_kickstarter_analysis(pledge_value: float,
                              goal_value: float,
                              category_value: str,
                              thisData: pd.DataFrame,
                              model):
    pledge_percentile = (thisData['averagePledge'] < pledge_value).mean()*100
    goal_percentile = (thisData['goal_usd'] < goal_value).mean()*100

    new_data = pd.DataFrame({
        'averagePledge':[pledge_value],
        'goal_usd':[goal_value],
        'parent_category':[category_value]
    })
    new_data['log_pledge'] = np.log10(new_data['averagePledge'])
    new_data['log_goal'] = np.log10(new_data['goal_usd'])

    pred = model.get_prediction(new_data)
    pred_mean = pred.predicted_mean[0]
    sigma = np.sqrt(model.mse_resid)

    # Probability of meeting goal
    z_val = (0 - pred_mean)/sigma
    prob_goal_is_met = 1 - norm.cdf(z_val)

    # Expected Revenue
    expectedRevenue = (10**pred_mean)*goal_value

    # Compute 90% CI for percent_of_goal_reached
    z_90 = 1.645
    lower_percent = 10**(pred_mean - z_90*sigma)
    upper_percent = 10**(pred_mean + z_90*sigma)

    # Compute percentile and interpretation for predicted percent_of_goal_reached
    percent_percentile, percent_interpretation = get_percentile_and_interpretation(10**pred_mean, thisData['percent_of_goal_reached'])

    # Compute 90% CI for expected revenue
    log_expected_rev_mean = np.log10(expectedRevenue)
    lower_rev = 10**(log_expected_rev_mean - z_90*sigma)
    upper_rev = 10**(log_expected_rev_mean + z_90*sigma)

    # Compute percentile and interpretation for expected revenue
    revenue_data = thisData['percent_of_goal_reached'] * thisData['goal_usd']
    revenue_percentile, revenue_interpretation = get_percentile_and_interpretation(expectedRevenue, revenue_data)

    # Average Pledge Plot
    fig_pledge, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(np.log10(thisData['averagePledge']), fill=True, color='#87CEEB', alpha=0.5, ax=ax)
    ax.axvline(np.log10(pledge_value), color='#87CEEB', linestyle='--', linewidth=2)
    ax.set_title("Average Pledge")
    ax.set_xlabel("Avg Pledge ($'s)")
    ax.set_ylabel("")
    ax.set_xticks(np.log10([10,100,1000,10000,100000]))
    ax.set_xticklabels(["$10", "$100", "$1k", "$10k", "$100k"])
    fig_pledge.tight_layout()

    # Goal Plot
    fig_goal, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(np.log10(thisData['goal_usd']), fill=True, color='#8fbc8f', alpha=0.5, ax=ax)
    ax.axvline(np.log10(goal_value), color='#8fbc8f', linestyle='--', linewidth=2)
    ax.set_title("Goal")
    ax.set_xlabel("Goal ($'s)")
    ax.set_ylabel("")
    ax.set_xticks(np.log10([10,100,1000,10000,100000,1000000]))
    ax.set_xticklabels(["$10", "$100", "$1k", "$10k", "$100k", "$1m"])
    fig_goal.tight_layout()

    # Percent of Goal Reached Plot
    fig_percent_temp, ax_temp = plt.subplots()
    sns.kdeplot(np.log10(thisData['percent_of_goal_reached'][thisData['percent_of_goal_reached']>0]), fill=False, ax=ax_temp)
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

    # Revenue Plot
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


# ------------------- MAIN APP -------------------
st.title("Kickstarter Analysis")

data = load_data()
data = preprocess_data(data)

st.sidebar.header("User Inputs")

with st.sidebar.expander("Pledge Value ($)"):
    st.write("This is the average amount each backer pledges to your campaign. Higher pledge values mean fewer backers are needed to reach your goal.")
pledge_value = st.sidebar.number_input("Pledge Value ($)", min_value=1.0, value=10.0, step=1.0)

with st.sidebar.expander("Unit Cost ($)"):
    st.write("This is the cost to you per unit (e.g., per reward item) that you produce and ship to a backer.")
unit_cost = st.sidebar.number_input("Unit Cost ($)", min_value=0.1, value=2.0, step=0.1)

with st.sidebar.expander("Overhead ($)"):
    st.write("These are the fixed costs you incur regardless of how many units you produce.")
overhead = st.sidebar.number_input("Overhead ($)", min_value=100.0, value=3000.0, step=100.0)

with st.sidebar.expander("Break Even % of Goal"):
    st.write("This is the fraction of your goal at which you break even.")
breakEvenPercentageOfGoal = st.sidebar.slider("Break Even % of Goal", min_value=0.01, max_value=1.0, value=0.95, step=0.01)

with st.sidebar.expander("Select Parent Category"):
    st.write("Choose the category of Kickstarter projects similar to yours.")
all_categories = data['parent_category'].dropna().unique().tolist()
selected_category = st.sidebar.selectbox("Select Parent Category:", options=all_categories, 
                                         index=all_categories.index("Art") if "Art" in all_categories else 0)

userData = filter_category(data, userCategory=selected_category)
userData, userBreakEvenQuantity, computed_goal_value, userProfit, userOverhead = compute_business_variables(
    userData, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal
)

override_goal = st.sidebar.radio("Enter own goal", ["Use computed goal", "Override goal"])
if override_goal == "Override goal":
    new_goal_value = st.sidebar.number_input("New Goal Value ($)", min_value=100.0, value=float(computed_goal_value), step=100.0)
else:
    new_goal_value = computed_goal_value

interaction_model_category = fit_interaction_model(data)

(fig_pledge, fig_goal, fig_percent, fig_revenue,
 pledge_percentile, goal_percentile, prob_goal_is_met, expectedRevenue,
 pred_mean, lower_percent, upper_percent, percent_percentile, percent_interpretation,
 lower_rev, upper_rev, revenue_percentile, revenue_interpretation) = plot_kickstarter_analysis(
     pledge_value, new_goal_value, category_value=selected_category, thisData=data, model=interaction_model_category
)

# Move Average Pledge section to the top
st.header("Average Pledge")
colA, colB = st.columns([2,1])
with colA:
    st.pyplot(fig_pledge)
with colB:
    # Show interpretation for Average Pledge
    pct_pledge, interp_pledge = get_percentile_and_interpretation(pledge_value, data['averagePledge'])
    st.markdown("<div style='text-align:center; font-size:24px;'>Average Pledge</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Pledge is at {pct_pledge:.2f}% percentile - {interp_pledge}</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Move the description/stats for Average Pledge above Business Variables
break_fig, overhead_fig, profit_fig = plot_business_variables(userData, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal)

st.markdown("<br><br>", unsafe_allow_html=True)

# # of Backers Section
st.subheader("# of Backers to Break Even")
st.latex(r"\#\text{ of Backers}_{\text{break-even}} = \frac{\text{Goal}_{\text{USD}}}{\text{Average Pledge}} \times \text{BreakEven\%}")
col_b1, col_b2 = st.columns([2,1])
with col_b1:
    st.pyplot(break_fig)
with col_b2:
    pct_backers, interpretation_backers = get_percentile_and_interpretation(userBreakEvenQuantity, userData['breakQ'])
    st.markdown(f"**Percentile:** {pct_backers:.2f}% - {interpretation_backers}")

st.markdown("<br><br>", unsafe_allow_html=True)

# Overhead Costs Section
st.subheader("Overhead Costs")
st.latex(r"\text{Overhead} = (\#\text{Backers}_{\text{break-even}}) \times (\text{Pledge} - \text{Unit Cost})")
col_o1, col_o2 = st.columns([2,1])
with col_o1:
    st.pyplot(overhead_fig)
with col_o2:
    pct_overhead_val, interpretation_overhead_val = get_percentile_and_interpretation(userOverhead, userData['fCost'])
    st.markdown(f"**Percentile:** {pct_overhead_val:.2f}% - {interpretation_overhead_val}")

st.markdown("<br><br>", unsafe_allow_html=True)

# Profit Section
st.subheader("Profit")
st.latex(r"\text{Profit} = (1 - \text{BreakEven\%}) \times \text{Goal}_{\text{USD}} \times \left(1 - \frac{\text{Unit Cost}}{\text{Pledge}}\right)")
col_p1, col_p2 = st.columns([2,1])
with col_p1:
    st.pyplot(profit_fig)
with col_p2:
    pct_profit_val, interpretation_profit_val = get_percentile_and_interpretation(userProfit, userData['profit'])
    st.markdown(f"**Percentile:** {pct_profit_val:.2f}% - {interpretation_profit_val}")

st.markdown("<br><br>", unsafe_allow_html=True)

st.header("Your Kickstarter Goal")
st.markdown(f"<div style='text-align:center; font-size:48px; font-weight:bold;'>{new_goal_value:,.2f}</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:20px;'>Recommended Goal</div>", unsafe_allow_html=True)

fig_goal_compare, ax = plt.subplots(figsize=(8,6))
sns.kdeplot(np.log10(userData['goal_usd']), fill=True, color='#8fbc8f', alpha=0.5, ax=ax)
ax.axvline(np.log10(new_goal_value), color='#8fbc8f', linestyle='--', linewidth=2)
ax.set_title("")
ax.set_xlabel("Goal ($)")
ax.set_ylabel("")
ax.set_xticks(np.log10([10,100,1000,10000,100000,1000000]))
ax.set_xticklabels(["$10", "$100", "$1k", "$10k", "$100k", "$1m"])
fig_goal_compare.tight_layout()

goal_percentile_val, goal_interpretation = get_percentile_and_interpretation(new_goal_value, data['goal_usd'])

col_g1, col_g2 = st.columns([2,1])
with col_g1:
    st.pyplot(fig_goal_compare)
with col_g2:
    st.markdown(f"**Percentile:** {goal_percentile_val:.2f}% - {goal_interpretation}")

st.markdown("<br><br>", unsafe_allow_html=True)

# Predicting chances of success section
st.subheader("Predicting the Chances of Success")

kick = filter_category(data, selected_category).copy()
kick['color'] = np.where(kick['passed'] == 1, "#90EE90", "#FFB6C1")
kick['label'] = np.where(kick['passed'] == 1, "Passing their goal", "Failed their goal")

fig_3d = go.Figure(data=go.Scatter3d(
    x=np.log10(kick['averagePledge']),
    y=np.log10(kick['goal_usd']),
    z=np.log10(kick['percent_of_goal_reached']),
    mode='markers',
    marker=dict(
        size=3,
        opacity=0.3,
        color=kick['color'],
    ),
    text=kick['label'],
    hovertemplate="<b>%{text}</b><br>Avg Pledge: %{x}<br>Goal: %{y}<br>%Goal: %{z}<extra></extra>"
))

fig_3d.update_layout(
    height=800,  # Increase the height of the Plotly graph
    scene=dict(
        zaxis=dict(
            title="% of goal reached",
            tickvals=np.log10([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]),
            ticktext=["0.1%", "1%","10%", "100%", "10x", "100x", "1,000x", "10,000x"]
        ),
        yaxis=dict(
            title="goal $'s",
            tickvals=np.log10([1, 10, 100, 1000, 10000, 100000]),
            ticktext=["$1", "$10", "$100", "$1k", "$10k", "$100k"]
        ),
        xaxis=dict(
            title="average pledge $'s",
            tickvals=np.log10([1, 10, 100, 1000, 10000, 100000]),
            ticktext=["$1", "$10", "$100", "$1k", "$10k", "$100k"]
        ),
        aspectmode="cube"
    )
)

st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Percent of Goal Reached
st.subheader("Percent of Goal Reached")
colE, colF = st.columns([2,1])
with colE:
    st.pyplot(fig_percent)
with colF:
    st.markdown("<div style='text-align:center; font-size:24px;'>Percent of Goal Reached</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Probability of meeting the goal: {prob_goal_is_met:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Predicted % of Goal: {100*(10**(pred_mean)):.2f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Percentile: {percent_percentile:.2f}% - {percent_interpretation}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>90% CI: [{100*lower_percent:.2f}%, {100*upper_percent:.2f}%]</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Expected Revenue
st.subheader("Expected Revenue")
colG, colH = st.columns([2,1])
with colG:
    st.pyplot(fig_revenue)
with colH:
    st.markdown("<div style='text-align:center; font-size:24px;'>Expected Revenue</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Expected Revenue: ${expectedRevenue:,.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Percentile: {revenue_percentile:.2f}% - {revenue_interpretation}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>90% CI: [${lower_rev:,.2f}, ${upper_rev:,.2f}]</div>", unsafe_allow_html=True)
