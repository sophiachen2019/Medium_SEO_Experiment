import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Medium SEO Experiment", layout="wide")

# --- CONFIGURATION ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/1UaDvWG5AGDlqA991Jg0Z7XBHNjqrdtsmVQpZazt6tx4/export?format=csv&gid=0"

# --- DATA LOADING ---
@st.cache_data(ttl=300) # Cache for 5 mins
def load_data():
    try:
        df = pd.read_csv(SHEET_URL)
        
        # Clean Headers
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        
        # Parse Dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']) # Drop empty dates
        df['date'] = df['date'].dt.date
        
        # Clean Numeric Columns
        def clean_num(val):
            if pd.isna(val): return 0
            s = str(val).replace(",", "").replace("$", "").replace("%", "").strip()
            if not s: return 0
            if '.' in s: return float(s)
            return int(s)

        cols_to_clean = ['views_total', 'reads_total', 'earnings_total', 'views_google']
        for col in cols_to_clean:
            if col in df.columns:
                 df[col] = df[col].apply(clean_num)

        # Filter out rows with 0 total views (Placeholder Future Dates)
        if 'views_total' in df.columns:
            df = df[df['views_total'] > 0]

        # Split into Daily and History
        # Logic: Before Start Date is History, On/After is Daily (Experiment)
        EXPERIMENT_START_DATE = "2026-01-28" 
        cutoff_date = pd.to_datetime(EXPERIMENT_START_DATE).date()
        
        history_df = df[df['date'] < cutoff_date].copy()
        daily_df = df[df['date'] >= cutoff_date].copy()
        
        # Renaissance mapping for standard columns expected by dashboard logic
        
        # Ensure is_treated is boolean
        # Robust conversion: 1, "1", True -> True. 0, "0", False -> False
        def to_bool(x):
            try:
                if isinstance(x, str):
                    x = x.lower().strip()
                    if x in ['true', '1', 'yes']: return True
                    return False
                return bool(x)
            except:
                return False

        if 'is_treated' in df.columns:
             daily_df['is_treated'] = daily_df['is_treated'].apply(to_bool)
             history_df['is_treated'] = history_df['is_treated'].apply(to_bool)
             
        # History DF needs 'snapshot_date' column for existing logic
        history_df['snapshot_date'] = history_df['date']
        
        # DEBUG: Sidebar Stats
        c_count = len(daily_df[daily_df['is_treated'] == False])
        t_count = len(daily_df[daily_df['is_treated'] == True])
        # st.sidebar.write(f"Debug: Treated Rows: {t_count}")
        # st.sidebar.write(f"Debug: Control Rows: {c_count}")
        
        # if c_count == 0:
        #     st.sidebar.warning("WARN: 0 Control Rows found!")

        return daily_df, history_df

    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return None, None

daily_df, history_df = load_data()

if daily_df is None or history_df is None:
    st.stop()

# --- PREPROCESSING ---
daily_df['date'] = pd.to_datetime(daily_df['date'])
treated_daily = daily_df[daily_df['is_treated'] == True].copy()
control_daily = daily_df[daily_df['is_treated'] == False].copy()

# --- DASHBOARD HEADER ---
st.title("🧪 Medium SEO Experiment: Product Sense")
st.markdown("Tracking the impact of metadata changes vs. Synthetic Control Group.")

# --- EXPERIMENT DETAILS ---
with st.expander("Experiment Hypothesis & Design", expanded=False):
    st.markdown("""
    **[Hypothesis]**
    By aligning the SEO metadata with high-intent search queries (e.g., 'Product Sense Interview Framework'), we will increase external traffic from search engines, leading to higher article views, reads and earnings.

    **[Experiment Design]**
    Since we cannot run a true A/B test (Medium doesn't allow two versions of the same URL), we will use a synthetic control causal inference approach.
    
    *   **Treatment**: Update SEO Title and Description for the [“Product Sense” article](https://medium.com/data-science-collective/demystifying-product-sense-in-data-scientist-interviews-ba25b3bc0cd0), per Gemini’s suggestion.
        *   *New Title*: Product Sense Interview Guide for Data Scientists (2026 Edition)
        *   *New Description*: Master the data science product sense interview with proven frameworks for metrics, product thinking, and case studies. Perfect for FAANG interview prep.
    *   **Control**: Monitor the [Analytics Projects](https://medium.com/data-science-collective/analytics-projects-in-data-science-a-practical-framework-09e2f5ec9317), [Tiered Metrics](https://medium.com/data-science-collective/designing-effective-metrics-cfe6b88c7a32) and [CLTV Modeling](https://medium.com/data-science-collective/from-data-to-decisions-a-practical-framework-to-quantifying-business-impact-369d1ae7f1cd) articles as a baseline.
        *   *Rationale*: Same publication (Data Science Collective), similar topic (Product Strategy & Metrics). This synthetic control group helps filter out platform-wide traffic dips or surges.
    *   **Duration**: 1/28/2026 - 2/10/2026
    """)

# --- METRICS CALCULATIONS ---

# 1. Slope Acceleration (Velocity)
st.subheader("Velocity Analysis")

def calculate_velocity(df, metric, lookback_days=None):
    if df.empty: return 0
    # Simple slope: (End - Start) / Days
    start_val = df[metric].iloc[0]
    end_val = df[metric].iloc[-1]
    days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    if days == 0: return 0
    return (end_val - start_val) / days

# --- HELPER FUNCTIONS ---
def get_historical_velocity(history_df, daily_treat_df, metric):
    # Filter treated history
    t_hist = history_df[history_df['is_treated'] == True].sort_values('snapshot_date')
    
    if len(t_hist) >= 2:
        start = t_hist.iloc[0]
        end = t_hist.iloc[-1]
        days = (end['snapshot_date'] - start['snapshot_date']).days
        if days > 0:
            return (end[metric] - start[metric]) / days
    elif len(t_hist) == 1 and not daily_treat_df.empty:
        start = t_hist.iloc[0]
        day0 = daily_treat_df.iloc[0]
        # ensure dates
        h_date = pd.to_datetime(start['snapshot_date'])
        d_date = pd.to_datetime(day0['date'])
        days = (d_date - h_date).days
        if days > 0:
            return (day0[metric] - start[metric]) / days
            
    return 0.0

def get_control_acceleration(history_df, daily_df, metric):
    # Control Daily & History
    c_daily = daily_df[daily_df['is_treated'] == False].copy()
    c_hist = history_df[history_df['is_treated'] == False].copy()
    
    ids = c_daily['post_id'].unique()
    
    hist_vels = []
    curr_vels = []
    
    for pid in ids:
        # Hist Vel
        h_sub = c_hist[c_hist['post_id'] == pid].sort_values('snapshot_date')
        d_sub = c_daily[c_daily['post_id'] == pid].sort_values('date')
        
        # Inline simplified logic
        h_val = 0.0
        if len(h_sub) >= 2:
            days = (h_sub.iloc[-1]['snapshot_date'] - h_sub.iloc[0]['snapshot_date']).days
            if days > 0: h_val = (h_sub.iloc[-1][metric] - h_sub.iloc[0][metric]) / days
        elif len(h_sub) == 1 and not d_sub.empty:
             days = (pd.to_datetime(d_sub.iloc[0]['date']) - pd.to_datetime(h_sub.iloc[0]['snapshot_date'])).days
             if days > 0: h_val = (d_sub.iloc[0][metric] - h_sub.iloc[0][metric]) / days
        
        hist_vels.append(h_val)
        
        # Curr Vel
        curr_vels.append(calculate_velocity(d_sub, metric))

    avg_hist = sum(hist_vels) / len(hist_vels) if hist_vels else 0
    avg_curr = sum(curr_vels) / len(curr_vels) if curr_vels else 0
    
    if avg_hist > 0:
        return (avg_curr / avg_hist - 1) * 100
    return 0.0

def plot_velocity_chart(daily_df, hist_vel, ctrl_accel, metric, title, label_map):
    if daily_df.empty: return None
    
    chart_df = daily_df[['date', metric]].copy()
    start_date = chart_df['date'].iloc[0]
    start_val = chart_df[metric].iloc[0]
    
    chart_df['days_since'] = (pd.to_datetime(chart_df['date']) - pd.to_datetime(start_date)).dt.days
    
    # Projections
    chart_df['Projected Baseline (History)'] = start_val + (chart_df['days_since'] * hist_vel)
    adj_vel = hist_vel * (1 + (ctrl_accel / 100))
    chart_df['Counterfactual (Control Adjusted)'] = start_val + (chart_df['days_since'] * adj_vel)
    
    # Format Date for Categorical Axis (Remove Time)
    if not pd.api.types.is_string_dtype(chart_df['date']):
         chart_df['date'] = chart_df['date'].dt.strftime('%Y-%m-%d')
         
    fig = px.line(chart_df, x='date', y=[metric, 'Projected Baseline (History)', 'Counterfactual (Control Adjusted)'],
                  title=title, labels=label_map, markers=True)
    fig.update_xaxes(type='category')
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector={"legendgroup": "Projected Baseline (History)"})
    fig.update_traces(patch={"line": {"dash": "dash", "color": "gray"}}, selector={"legendgroup": "Counterfactual (Control Adjusted)"})
    fig.update_traces(patch={"line": {"width": 3}}, selector={"legendgroup": metric})
    return fig

# --- METRIC CALCULATIONS ---

# 1. Google Views
hist_vel_google = get_historical_velocity(history_df, treated_daily, 'views_google')
curr_vel_google = calculate_velocity(treated_daily, 'views_google')
ctrl_accel_google = get_control_acceleration(history_df, daily_df, 'views_google')
accel_google = 0.0
if hist_vel_google > 0: accel_google = (curr_vel_google / hist_vel_google - 1) * 100

# 2. Total Views
hist_vel_total = get_historical_velocity(history_df, treated_daily, 'views_total')
curr_vel_total = calculate_velocity(treated_daily, 'views_total')
ctrl_accel_total = get_control_acceleration(history_df, daily_df, 'views_total')
accel_total = 0.0
if hist_vel_total > 0: accel_total = (curr_vel_total / hist_vel_total - 1) * 100

# --- VISUALIZATION ---

# Metrics Summary Section
st.markdown("#### Total Views")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Hist. Velocity", f"{hist_vel_total:.2f}/day")
col2.metric("Curr. Velocity", f"{curr_vel_total:.2f}/day", delta=f"{curr_vel_total - hist_vel_total:.2f}")
col3.metric("Acceleration", f"{accel_total:.1f}%")
col4.metric("Control Accel", f"{ctrl_accel_total:.1f}%", delta=f"{accel_total - ctrl_accel_total:.1f}% (Net)")

st.markdown("#### Google Search Views")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Hist. Velocity", f"{hist_vel_google:.2f}/day")
col2.metric("Curr. Velocity", f"{curr_vel_google:.2f}/day", delta=f"{curr_vel_google - hist_vel_google:.2f}")
col3.metric("Acceleration", f"{accel_google:.1f}%")
col4.metric("Control Accel", f"{ctrl_accel_google:.1f}%", delta=f"{accel_google - ctrl_accel_google:.1f}% (Net)")

# Charts Section
st.markdown("### Velocity Charts")

fig_total = plot_velocity_chart(
    treated_daily, hist_vel_total, ctrl_accel_total, 'views_total', 
    "Velocity: Total Views (Actual vs Projected)",
    {'views_total': 'Total Views', 'value': 'Views', 'variable': 'Trend'}
)
if fig_total: st.plotly_chart(fig_total, use_container_width=True)

fig_google = plot_velocity_chart(
    treated_daily, hist_vel_google, ctrl_accel_google, 'views_google', 
    "Velocity: Google Search Views (Actual vs Projected)",
    {'views_google': 'Google Search Views', 'value': 'Views', 'variable': 'Trend'}
)
if fig_google: st.plotly_chart(fig_google, use_container_width=True)
 
st.markdown("""
**Interpretation:**
*   **Actual Views**: The real performance of your article.
*   **Projected Baseline**: Where you would be if you just kept your *old* velocity (0% acceleration).
*   **Counterfactual**: Where you would be if you accelerated at the *same rate as the Control Group*. 
""")

# --- CHARTS ---

# 2. Synthetic Control Lift (Indexed)
st.subheader("Indexed Growth of Views")

# Normalize Function: Index to Day 0 = 100
def normalize_to_start(df, metric):
    if df.empty: return df
    start_val = df[metric].iloc[0]
    
    col_name = f'{metric}_indexed'
    if start_val == 0: 
        # Decide how to handle 0 start. 
        # Option A: Set all to 100? No. 
        # Option B: Set to 0? Yes.
        df[col_name] = 0.0
    else:
        df[col_name] = (df[metric] / start_val) * 100
    return df

# Index Treated
from numpy import inf

treated_daily = treated_daily.sort_values('date') # Ensure sorted
treated_daily = normalize_to_start(treated_daily, 'views_total')
treated_daily = normalize_to_start(treated_daily, 'earnings_total')
treated_daily = normalize_to_start(treated_daily, 'views_google') # Added

# Index Controls & Average
control_ids = control_daily['post_id'].unique()
control_ids = control_daily['post_id'].unique()
control_indices = []

for pid in control_ids:
    c_df = control_daily[control_daily['post_id'] == pid].copy().sort_values('date')
    c_df = normalize_to_start(c_df, 'views_total')
    c_df = normalize_to_start(c_df, 'earnings_total')
    c_df = normalize_to_start(c_df, 'views_google')
    control_indices.append(c_df[['date', 'views_total_indexed', 'earnings_total_indexed', 'views_google_indexed']].set_index('date'))

if control_indices:
    # Concatenate all control DFs
    all_controls = pd.concat(control_indices)
    
    # Aggregation: Mean, Min, Max
    # Groups by Date (index)
    control_stats = all_controls.groupby(level=0).agg(['mean', 'min', 'max'])
    
    # Flatten MultiIndex columns (e.g., ('views_total_indexed', 'mean') -> 'views_total_indexed_mean')
    control_stats.columns = ['_'.join(col).strip() for col in control_stats.columns.values]
    
    # Reset index to make 'date' a column
    control_stats = control_stats.reset_index()
    
    # Rename for clarity interacting with comparison_df logic
    # We mostly need the _mean for the main line, and _min/_max for bands
    rename_map = {
        'views_total_indexed_mean': 'views_total_indexed_control_avg',
        'views_total_indexed_min': 'views_total_indexed_control_min',
        'views_total_indexed_max': 'views_total_indexed_control_max',
        
        'earnings_total_indexed_mean': 'earnings_total_indexed_control_avg',
        'earnings_total_indexed_min': 'earnings_total_indexed_control_min',
        'earnings_total_indexed_max': 'earnings_total_indexed_control_max',
        
        'views_google_indexed_mean': 'views_google_indexed_control_avg',
        'views_google_indexed_min': 'views_google_indexed_control_min',
        'views_google_indexed_max': 'views_google_indexed_control_max',
    }
    control_stats = control_stats.rename(columns=rename_map)

    # Init treated for merge with explicit names
    treated_for_merge = treated_daily[['date', 'views_total_indexed', 'earnings_total_indexed', 'views_google_indexed']].copy()
    treated_for_merge = treated_for_merge.rename(columns={
        'views_total_indexed': 'views_total_indexed_treated',
        'earnings_total_indexed': 'earnings_total_indexed_treated',
        'views_google_indexed': 'views_google_indexed_treated'
    })

    # Merge Treated vs Control Stats
    comparison_df = pd.merge(
        treated_for_merge, 
        control_stats, 
        on='date', 
        how='outer'
    )
    
else:
    # No controls found, just plot Treated
    st.warning("No Control Group data found. Showing Treated only.")
    comparison_df = treated_daily.copy()
    comparison_df = comparison_df.rename(columns={
        'views_total_indexed': 'views_total_indexed_treated',
        'views_google_indexed': 'views_google_indexed_treated'
    })
    # Add dummy control columns
    for col in ['views_total_indexed_control_avg', 'views_total_indexed_control_min', 'views_total_indexed_control_max',
                'views_google_indexed_control_avg', 'views_google_indexed_control_min', 'views_google_indexed_control_max']:
        comparison_df[col] = None

if comparison_df.empty:
    st.warning("Comparison DF is empty! Check date alignment.")

# Calculate Lift for Total Views
comparison_df['Lift_Views'] = comparison_df['views_total_indexed_treated'] / comparison_df['views_total_indexed_control_avg']

# Calculate Lift for Google Views
if 'views_google_indexed_treated' in comparison_df.columns:
     comparison_df['Lift_Google'] = comparison_df['views_google_indexed_treated'] / comparison_df['views_google_indexed_control_avg']


# --- PREPARE DETAILED DATA (For Detail Charts) ---
all_indexed_frames = []
unique_ids = daily_df['post_id'].unique()

for pid in unique_ids:
    sub_df = daily_df[daily_df['post_id'] == pid].copy().sort_values('date')
    if not sub_df.empty:
        # Calculate Indexed Metrics
        sub_df = normalize_to_start(sub_df, 'views_total')
        if 'views_google' in sub_df.columns:
            sub_df = normalize_to_start(sub_df, 'views_google')
        all_indexed_frames.append(sub_df)

viz_df = pd.DataFrame()
color_col = 'post_id'
if all_indexed_frames:
    viz_df = pd.concat(all_indexed_frames)
    
    # Format Date for X-Axis
    if not pd.api.types.is_string_dtype(viz_df['date']):
        viz_df['date'] = pd.to_datetime(viz_df['date']).dt.strftime('%Y-%m-%d')
    
    # Determine Label Column (Topic > Title > ID)
    if 'topic' in viz_df.columns:
        color_col = 'topic'
    elif 'title' in viz_df.columns:
        color_col = 'title'

# --- HELPER FOR BANDS ---
def add_envelope(fig, df, x_col, min_col, max_col, name="Control Range"):
    # Add Lower Bound (Transparent)
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[min_col],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name=f"{name} Lower"
    ))
    # Add Upper Bound (Filled to Lower)
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[max_col],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(200, 200, 200, 0.3)', # Light Gray
        name=name,
        showlegend=True
    ))
    return fig

# Format Date for Categorical Axis (Remove Time)
# We need to ensure comparison_df['date'] is datetime before formatting if it isn't already
if not pd.api.types.is_string_dtype(comparison_df['date']):
    comparison_df['date'] = pd.to_datetime(comparison_df['date']).dt.strftime('%Y-%m-%d')

# Chart 1: Total Views Lift
fig_lift = px.line(comparison_df, x='date', y=['views_total_indexed_treated', 'views_total_indexed_control_avg'],
                   title="Total Views: Treated vs Control Range (Indexed, Day 0 = 100)",
                   color_discrete_map={'views_total_indexed_treated': 'blue', 'views_total_indexed_control_avg': 'orange'},
                   labels={'value': 'Indexed Growth', 'variable': 'Group'},
                   markers=True)
fig_lift.update_xaxes(type='category')

# Add Range
if 'views_total_indexed_control_min' in comparison_df.columns:
    fig_lift = add_envelope(fig_lift, comparison_df, 'date', 'views_total_indexed_control_min', 'views_total_indexed_control_max')

st.plotly_chart(fig_lift, use_container_width=True)

# Detailed Chart: Total Views
if not viz_df.empty:
    fig_all_total = px.line(viz_df, x='date', y='views_total_indexed', color=color_col,
                            title="Total Views Indexed Growth (By Article)", 
                            labels={'views_total_indexed': 'Indexed Growth (Day 0 = 100)', 'date': 'Date'},
                            markers=True)
    fig_all_total.update_xaxes(type='category')
    st.plotly_chart(fig_all_total, use_container_width=True)

# Chart 2: Google Search Views Lift
if 'views_google_indexed_treated' in comparison_df.columns:
    fig_google_lift = px.line(comparison_df, x='date', y=['views_google_indexed_treated', 'views_google_indexed_control_avg'],
                    title="Google Search Views: Treated vs Control Range (Indexed, Day 0 = 100)",
                    color_discrete_map={'views_google_indexed_treated': 'blue', 'views_google_indexed_control_avg': 'orange'},
                    labels={'value': 'Indexed Growth', 'variable': 'Group'},
                    markers=True)
    fig_google_lift.update_xaxes(type='category')
    
    # Add Range
    if 'views_google_indexed_control_min' in comparison_df.columns:
        fig_google_lift = add_envelope(fig_google_lift, comparison_df, 'date', 'views_google_indexed_control_min', 'views_google_indexed_control_max')

    st.plotly_chart(fig_google_lift, use_container_width=True)
    
    if 'Lift_Google' in comparison_df.columns and not pd.isna(comparison_df['Lift_Google'].iloc[-1]):
        last_google_lift = comparison_df['Lift_Google'].iloc[-1]
        st.metric("Google Search Lift (vs Control)", f"{last_google_lift:.2f}x")

    # Detailed Chart: Google Search Views
    if not viz_df.empty and 'views_google_indexed' in viz_df.columns:
        fig_all_google = px.line(viz_df, x='date', y='views_google_indexed', color=color_col,
                                title="Google Search Views Indexed Growth (By Article)", 
                                labels={'views_google_indexed': 'Indexed Growth (Day 0 = 100)', 'date': 'Date'},
                                markers=True)
        fig_all_google.update_xaxes(type='category')
        st.plotly_chart(fig_all_google, use_container_width=True)

st.markdown("""
**Interpretation:**
*   **Indexed Growth**: All articles start at 100 on Day 0.
*   **Control Range**: The shaded grey area represents the range of performance (min to max) of the control articles.
*   **Divergence**: If the "Treated" line goes **above** the "Control Range", your changes are outperforming the baseline.
*   **Lift Metric**: A value of 1.2x means you have 20% more views than expected based on the control group's behavior.
""")



# 3. Efficiency Metrics
st.subheader("Efficiency Analysis")

# Re-define treated_history for Efficiency Analysis (needed for hist_snapshot)
treated_history = history_df[history_df['is_treated'] == True].sort_values('snapshot_date')

# Prepare Combined Data for Plotting (History + Daily)
# We want the 9/28 point to be the start of the line
if not treated_history.empty and not treated_daily.empty:
    # 1. Get History Snapshot (just 9/28 or relevant start)
    hist_snapshot = treated_history.iloc[[0]].copy() # Keep as DataFrame
    
    # Ensure columns match for concat
    # We need: date, reads_total, earnings_total, views_total
    # history_df has 'snapshot_date' instead of 'date' usually, or we fixed it in load_data?
    # load_data says: history_df['snapshot_date'] = history_df['date']
    # So 'date' exists in history_df.
    
    cols_needed = ['date', 'reads_total', 'earnings_total', 'views_total']
    plot_df = pd.concat([hist_snapshot[cols_needed], treated_daily[cols_needed]])
    
    # Fix TypeError: Convert to explicit datetime before sorting
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values('date')
    
    # Convert back to string for Categorical Plotting (removes visual gap)
    plot_df['date'] = plot_df['date'].dt.strftime('%Y-%m-%d')
    
    # Calculate Efficiency Metrics on Projected DF
    plot_df['epv'] = plot_df.apply(lambda row: row['earnings_total'] / row['views_total'] if row['views_total'] > 0 else 0, axis=1)
    plot_df['read_ratio'] = plot_df.apply(lambda row: row['reads_total'] / row['views_total'] if row['views_total'] > 0 else 0, axis=1)

    # Row 1: Reads & Earnings
    eff_row1_col1, eff_row1_col2 = st.columns(2)

    with eff_row1_col1:
        fig_reads = px.line(plot_df, x='date', y='reads_total', title="Cumulative Reads", markers=True)
        fig_reads.update_xaxes(type='category')
        st.plotly_chart(fig_reads, use_container_width=True)
        
    with eff_row1_col2:
        fig_earn = px.line(plot_df, x='date', y='earnings_total', title="Cumulative Earnings ($)", markers=True)
        fig_earn.update_xaxes(type='category')
        st.plotly_chart(fig_earn, use_container_width=True)

    # Row 2: EPV & Read Ratio
    eff_row2_col1, eff_row2_col2 = st.columns(2)
    
    with eff_row2_col1:
        fig_epv = px.line(plot_df, x='date', y='epv', title="Earnings Per View (EPV)", markers=True)
        fig_epv.update_xaxes(type='category')
        st.plotly_chart(fig_epv, use_container_width=True)
        st.caption("Money earned per unique view. Higher is better.")
        
    with eff_row2_col2:
        fig_rr = px.line(plot_df, x='date', y='read_ratio', title="Read Ratio", markers=True)
        fig_rr.update_xaxes(type='category')
        st.plotly_chart(fig_rr, use_container_width=True)
        st.caption("Reads / Views. Measures content engagement quality.")
    
    st.markdown("""
    **Interpretation:**
    *   **Charts**: Show the trend starting from the Historical Baseline (9/28/2025) through the active experiment period.
    *   **Gap**: The straight line connects the 2025 baseline directly to the 2026 experiment start, visually showing the jump (or drop) in performance over the intervening months.
    """)
else:
    st.warning("Insufficient data for Efficiency Analysis (Missing history or daily data)")

# --- RAW DATA ---
with st.expander("Raw Data"):
    st.write("Treated Daily", treated_daily)
    st.write("Control Daily", control_daily)
    st.write("Historical Baseline", history_df)
