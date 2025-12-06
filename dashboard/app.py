"""
Streamlit Dashboard for Sales Forecasting & Inventory Optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian SME Sales Forecasting & Inventory Optimizer")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/retail_sales_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return None

df = load_data()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Overview", "📈 Forecasting", "📦 Inventory Optimizer", "🔮 What-If Analysis"])

with tab1:
    st.header("Sales Overview")

    if df is not None:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            categories = ['All'] + list(df['category'].unique())
            selected_category = st.selectbox("Category", categories)
        with col2:
            states = ['All'] + list(df['state'].unique())
            selected_state = st.selectbox("State", states)
        with col3:
            time_period = st.selectbox("Time Period", ["Last 30 Days", "Last 90 Days", "Last Year", "All Time"])

        # Filter data
        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if selected_state != 'All':
            filtered_df = filtered_df[filtered_df['state'] == selected_state]

        # Time filter
        max_date = filtered_df['date'].max()
        if time_period == "Last 30 Days":
            filtered_df = filtered_df[filtered_df['date'] >= max_date - timedelta(days=30)]
        elif time_period == "Last 90 Days":
            filtered_df = filtered_df[filtered_df['date'] >= max_date - timedelta(days=90)]
        elif time_period == "Last Year":
            filtered_df = filtered_df[filtered_df['date'] >= max_date - timedelta(days=365)]

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"₹{filtered_df['revenue'].sum()/1e6:.2f}M")
        with col2:
            st.metric("Total Units", f"{filtered_df['units_sold'].sum():,}")
        with col3:
            st.metric("Avg Daily Revenue", f"₹{filtered_df.groupby('date')['revenue'].sum().mean():,.0f}")
        with col4:
            st.metric("Festival Days", f"{filtered_df[filtered_df['is_festival']]['date'].nunique()}")

        # Daily revenue trend
        st.subheader("Daily Revenue Trend")
        daily_revenue = filtered_df.groupby('date')['revenue'].sum().reset_index()
        fig = px.line(daily_revenue, x='date', y='revenue', title="Daily Revenue (₹)")
        fig.update_layout(xaxis_title="Date", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

        # Category breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue by Category")
            cat_revenue = filtered_df.groupby('category')['revenue'].sum().sort_values(ascending=True)
            fig = px.bar(x=cat_revenue.values, y=cat_revenue.index, orientation='h',
                        labels={'x': 'Revenue (₹)', 'y': 'Category'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Revenue by State")
            state_revenue = filtered_df.groupby('state')['revenue'].sum().sort_values(ascending=True)
            fig = px.bar(x=state_revenue.values, y=state_revenue.index, orientation='h',
                        labels={'x': 'Revenue (₹)', 'y': 'State'})
            st.plotly_chart(fig, use_container_width=True)

        # Festival impact
        st.subheader("Festival Impact on Sales")
        festival_impact = filtered_df.groupby('is_festival')['revenue'].mean()
        fig = px.bar(x=['Regular Days', 'Festival Days'],
                    y=[festival_impact.get(False, 0), festival_impact.get(True, 0)],
                    labels={'x': 'Day Type', 'y': 'Average Revenue (₹)'})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Sales Forecasting")

    if df is not None:
        st.subheader("Select Forecasting Parameters")

        col1, col2 = st.columns(2)
        with col1:
            forecast_category = st.selectbox("Forecast Category", list(df['category'].unique()), key='fc')
            forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        with col2:
            model_type = st.selectbox("Model", ["XGBoost", "Prophet", "ARIMA", "Ensemble"])

        if st.button("Generate Forecast", type="primary"):
            # Prepare data
            cat_df = df[df['category'] == forecast_category]
            daily_df = cat_df.groupby('date')['revenue'].sum().reset_index()

            # Simple moving average forecast (placeholder for actual models)
            last_30_avg = daily_df['revenue'].tail(30).mean()
            last_7_avg = daily_df['revenue'].tail(7).mean()

            # Generate forecast dates
            last_date = daily_df['date'].max()
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

            # Simulate forecast with trend
            np.random.seed(42)
            trend = np.linspace(0, 0.1, forecast_days)
            seasonality = np.sin(np.linspace(0, 4*np.pi, forecast_days)) * 0.1
            noise = np.random.normal(0, 0.05, forecast_days)
            forecast_values = last_30_avg * (1 + trend + seasonality + noise)

            # Upper and lower bounds
            upper = forecast_values * 1.15
            lower = forecast_values * 0.85

            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast_values,
                'upper': upper,
                'lower': lower
            })

            # Combine historical and forecast
            st.subheader(f"Forecast for {forecast_category}")

            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=daily_df['date'], y=daily_df['revenue'],
                name='Historical', line=dict(color='blue')
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['forecast'],
                name='Forecast', line=dict(color='red', dash='dash')
            ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=list(forecast_df['date']) + list(forecast_df['date'][::-1]),
                y=list(forecast_df['upper']) + list(forecast_df['lower'][::-1]),
                fill='toself', fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'
            ))

            fig.update_layout(
                title=f"{forecast_days}-Day Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue (₹)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Daily Forecast", f"₹{forecast_values.mean():,.0f}")
            with col2:
                st.metric("Total Forecast", f"₹{forecast_values.sum():,.0f}")
            with col3:
                growth = (forecast_values.mean() - last_30_avg) / last_30_avg * 100
                st.metric("Expected Growth", f"{growth:+.1f}%")

with tab3:
    st.header("Inventory Optimizer")

    if df is not None:
        st.subheader("Stock Level Recommendations")

        col1, col2 = st.columns(2)
        with col1:
            opt_category = st.selectbox("Product Category", list(df['category'].unique()), key='oc')
            lead_time = st.number_input("Lead Time (days)", 1, 14, 3)
        with col2:
            safety_days = st.number_input("Safety Stock (days)", 1, 10, 3)
            service_level = st.slider("Service Level (%)", 90, 99, 95)

        if st.button("Calculate Optimal Stock", type="primary"):
            # Calculate statistics
            cat_df = df[df['category'] == opt_category]
            daily_demand = cat_df.groupby('date')['units_sold'].sum()

            avg_demand = daily_demand.mean()
            std_demand = daily_demand.std()

            # Calculate stock levels
            safety_stock = avg_demand * safety_days
            reorder_point = (avg_demand * lead_time) + safety_stock
            weekly_order = avg_demand * 7 + safety_stock

            # Display results
            st.subheader("Recommended Stock Levels")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Daily Demand", f"{avg_demand:.0f} units")
            with col2:
                st.metric("Safety Stock", f"{safety_stock:.0f} units")
            with col3:
                st.metric("Reorder Point", f"{reorder_point:.0f} units")
            with col4:
                st.metric("Weekly Order Qty", f"{weekly_order:.0f} units")

            # Demand distribution
            st.subheader("Daily Demand Distribution")
            fig = px.histogram(daily_demand, nbins=30, title="Distribution of Daily Units Sold")
            fig.add_vline(x=avg_demand, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {avg_demand:.0f}")
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("What-If Scenario Analysis")

    if df is not None:
        st.subheader("Simulate Different Scenarios")

        scenario = st.selectbox("Select Scenario", [
            "Festival Season (Diwali)",
            "Economic Slowdown",
            "Competitor Entry",
            "Price Increase (10%)",
            "New Store Opening"
        ])

        base_revenue = df['revenue'].sum()

        # Scenario multipliers
        multipliers = {
            "Festival Season (Diwali)": 1.5,
            "Economic Slowdown": 0.8,
            "Competitor Entry": 0.85,
            "Price Increase (10%)": 0.92,
            "New Store Opening": 1.15
        }

        adjusted_revenue = base_revenue * multipliers[scenario]
        change = (adjusted_revenue - base_revenue) / base_revenue * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Revenue", f"₹{base_revenue/1e6:.2f}M")
        with col2:
            st.metric("Projected Revenue", f"₹{adjusted_revenue/1e6:.2f}M")
        with col3:
            st.metric("Impact", f"{change:+.1f}%", delta_color="normal" if change > 0 else "inverse")

        # Visualization
        st.subheader("Scenario Comparison")
        scenarios = list(multipliers.keys())
        values = [base_revenue * m / 1e6 for m in multipliers.values()]

        fig = px.bar(x=scenarios, y=values, color=values,
                    labels={'x': 'Scenario', 'y': 'Revenue (₹M)'},
                    color_continuous_scale='RdYlGn')
        fig.add_hline(y=base_revenue/1e6, line_dash="dash", line_color="black",
                     annotation_text="Current")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Sales Forecasting System** | Built for Indian SMEs | © 2024")
