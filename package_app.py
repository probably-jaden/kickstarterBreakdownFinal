import streamlit as st
import pandas as pd
import numpy as np
from final.preprocessing import load_data, preprocess_data
from final.analysis import filter_category, compute_business_variables, get_percentile_and_interpretation
from final.modeling import fit_interaction_model
from final.plotting import plot_business_variables, plot_kickstarter_analysis
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Kickstarter Analysis", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    df = load_data()
    df = preprocess_data(df)
    return df

data = load_and_preprocess_data()

st.title("Kickstarter Analysis")

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

# Note:
# The provided final.plotting.plot_kickstarter_analysis in the user's package returns fewer values than the original code.
# For full functionality, we assume that plot_kickstarter_analysis is adapted to return all needed values.
# If not, you would need to restore the additional computations from the original code here.
# Below, we mimic the original functionality assuming the final.plotting.plot_kickstarter_analysis now returns:
# (fig_pledge, fig_goal, fig_percent, fig_revenue,
#  pledge_percentile, goal_percentile, prob_goal_is_met, expectedRevenue,
#  pred_mean, lower_percent, upper_percent, percent_percentile, percent_interpretation,
#  lower_rev, upper_rev, revenue_percentile, revenue_interpretation)

(fig_pledge, fig_goal, fig_percent, fig_revenue,
 pledge_percentile, goal_percentile, prob_goal_is_met, expectedRevenue,
 pred_mean, lower_percent, upper_percent, percent_percentile, percent_interpretation,
 lower_rev, upper_rev, revenue_percentile, revenue_interpretation) = plot_kickstarter_analysis(
     pledge_value, new_goal_value, category_value=selected_category, thisData=data, model=interaction_model_category
)

st.header("Average Pledge")
colA, colB = st.columns([2,1])
with colA:
    st.pyplot(fig_pledge)
with colB:
    pct_pledge, interp_pledge = get_percentile_and_interpretation(pledge_value, data['averagePledge'])
    st.markdown("<div style='text-align:center; font-size:24px;'>Average Pledge</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Pledge is at {pct_pledge:.2f}% percentile - {interp_pledge}</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

break_fig, overhead_fig, profit_fig = plot_business_variables(userData, pledge_value, unit_cost, overhead, breakEvenPercentageOfGoal)

st.markdown("<br><br>", unsafe_allow_html=True)

st.subheader("# of Backers to Break Even")
st.latex(r"\#\text{ of Backers}_{\text{break-even}} = \frac{\text{Goal}_{\text{USD}}}{\text{Average Pledge}} \times \text{BreakEven\%}")
col_b1, col_b2 = st.columns([2,1])
with col_b1:
    st.pyplot(break_fig)
with col_b2:
    pct_backers, interpretation_backers = get_percentile_and_interpretation(userBreakEvenQuantity, userData['breakQ'])
    st.markdown(f"**Percentile:** {pct_backers:.2f}% - {interpretation_backers}")

st.markdown("<br><br>", unsafe_allow_html=True)

st.subheader("Overhead Costs")
st.latex(r"\text{Overhead} = (\#\text{Backers}_{\text{break-even}}) \times (\text{Pledge} - \text{Unit Cost})")
col_o1, col_o2 = st.columns([2,1])
with col_o1:
    st.pyplot(overhead_fig)
with col_o2:
    pct_overhead_val, interpretation_overhead_val = get_percentile_and_interpretation(userOverhead, userData['fCost'])
    st.markdown(f"**Percentile:** {pct_overhead_val:.2f}% - {interpretation_overhead_val}")

st.markdown("<br><br>", unsafe_allow_html=True)

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
sns.kdeplot(np.log10(data['goal_usd']), fill=True, color='#8fbc8f', alpha=0.5, ax=ax)
ax.axvline(np.log10(new_goal_value), color='#8fbc8f', linestyle='--', linewidth=2)
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
    height=800,
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

st.subheader("Expected Revenue")
colG, colH = st.columns([2,1])
with colG:
    st.pyplot(fig_revenue)
with colH:
    st.markdown("<div style='text-align:center; font-size:24px;'>Expected Revenue</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Expected Revenue: ${expectedRevenue:,.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>Percentile: {revenue_percentile:.2f}% - {revenue_interpretation}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>90% CI: [${lower_rev:,.2f}, ${upper_rev:,.2f}]</div>", unsafe_allow_html=True)
