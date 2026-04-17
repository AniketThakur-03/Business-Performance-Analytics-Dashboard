from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_utils import (
    apply_filters,
    build_anomaly_table,
    build_customer_summary,
    build_data_quality_summary,
    build_executive_kpi_summary,
    build_forecast_frame,
    build_product_opportunity_table,
    build_rfm_summary,
    build_state_summary,
    load_superstore_data,
    simulate_discount_change,
)
from src.modeling import (
    build_discount_risk_segments,
    get_confusion_matrix_frame,
    get_feature_importance_table,
    get_regression_preview_frame,
    train_loss_classifier,
    train_profit_regressor,
)
from src.sql_store import (
    DB_PATH,
    DEFAULT_SQL_QUERIES,
    build_sqlite_store,
    get_table_counts,
    run_sql_query,
)

st.set_page_config(
    page_title="Business Performance Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.6rem;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 12px 14px;
    }

    div[data-testid="stMetric"] label {
        color: #9ca3af !important;
    }

    div[data-testid="stMetric"] div {
        color: #f9fafb !important;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
    }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.6rem;
        border: 1px solid #334155;
        border-radius: 999px;
        margin-right: 0.35rem;
        font-size: 0.85rem;
        color: #cbd5e1;
        background: #0f172a;
    }

    .status-card {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 14px 16px;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_superstore_data()


@st.cache_resource
def get_sql_store() -> str:
    df = get_data()
    build_sqlite_store(df)
    return str(DB_PATH)


@st.cache_resource
def get_models():
    base_df = get_data()
    return train_loss_classifier(base_df), train_profit_regressor(base_df)


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def get_date_column(df: pd.DataFrame) -> str | None:
    if "Order Date" in df.columns:
        return "Order Date"
    if "order_date" in df.columns:
        return "order_date"
    return None


def render_header(total_rows: int):
    left, right = st.columns([1.9, 1])

    with left:
        st.title("Business Performance Analytics Dashboard")
        st.write(
            "A business analytics dashboard built on the Superstore dataset to analyze "
            "sales, profit, customer trends, forecasting, and operational performance."
        )
        st.caption(f"Dataset rows loaded: {total_rows:,}")

   
def render_kpis(filtered: pd.DataFrame):
    summary = build_executive_kpi_summary(filtered)
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric("Revenue", f"${summary['total_sales'] / 1_000_000:.2f}M", pct(summary["sales_delta"]))
    m2.metric("Net Profit", f"${summary['total_profit'] / 1_000:.0f}K", pct(summary["profit_delta"]))
    m3.metric("Profit Margin", pct(summary["profit_margin"]))
    m4.metric("Orders", f"{int(summary['total_orders']):,}")
    m5.metric("Customers", f"{int(summary['total_customers']):,}")
    m6.metric("Loss-Making Orders", pct(summary["loss_rate"]))

    st.markdown(
        '<p class="small-note">These KPI cards compare the latest month against the previous month inside the current filtered view.</p>',
        unsafe_allow_html=True,
    )

def render_executive_tab(filtered: pd.DataFrame):
    st.subheader("Business Overview")
    st.caption("High-level view of revenue, profit, and regional performance.")

    forecast = build_forecast_frame(filtered, periods=4)
    anomalies = build_anomaly_table(filtered)

    left, right = st.columns([1.15, 0.85])

    with left:
        fig = go.Figure()

        history = forecast[forecast["type"] == "historical"]
        future = forecast[forecast["type"] == "forecast"]

        fig.add_trace(
            go.Scatter(
                x=history["order_month_ts"],
                y=history["sales"],
                mode="lines+markers",
                name="Historical Sales",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future["order_month_ts"],
                y=future["sales"],
                mode="lines+markers",
                name="Forecast Sales",
            )
        )
        fig.update_layout(
            title="Sales Trend with Forecast",
            xaxis_title="Month",
            yaxis_title="Sales",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Monthly Alert Summary**")
        alert_view = anomalies[anomalies["alert_flag"]].head(8)[
            ["order_month", "sales", "profit", "alert_reason"]
        ]
        if alert_view.empty:
            st.success("No unusually large monthly spikes or dips were detected.")
        else:
            st.dataframe(alert_view, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        region_summary = (
            filtered.groupby("Region", as_index=False)
            .agg(
                sales=("Sales", "sum"),
                profit=("Profit", "sum"),
                avg_discount=("Discount", "mean"),
            )
            .sort_values("sales", ascending=False)
        )
        fig = px.bar(
            region_summary,
            x="Region",
            y="sales",
            color="profit",
            hover_data=["avg_discount"],
            title="Regional Performance Snapshot",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        category_summary = (
            filtered.groupby("Category", as_index=False)
            .agg(
                sales=("Sales", "sum"),
                profit=("Profit", "sum"),
                orders=("Order ID", "nunique"),
            )
            .sort_values("sales", ascending=False)
        )
        fig = px.bar(
            category_summary,
            x="Category",
            y="profit",
            color="sales",
            title="Category Profit Contribution",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_commercial_tab(filtered: pd.DataFrame):
    st.subheader("Sales and Product Performance")
    st.caption("Focus on category, sub-category, and geographic sales performance.")

    opportunity = build_product_opportunity_table(filtered)
    subcat = (
        filtered.groupby("Sub-Category", as_index=False)
        .agg(
            sales=("Sales", "sum"),
            profit=("Profit", "sum"),
            avg_discount=("Discount", "mean"),
        )
    )

    c1, c2 = st.columns([1.05, 0.95])

    with c1:
        fig = px.scatter(
            subcat,
            x="avg_discount",
            y="profit",
            size="sales",
            text="Sub-Category",
            title="Sub-Category Profitability vs Discount Pressure",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top Product Opportunities**")
        st.dataframe(
            opportunity[
                ["Category", "Sub-Category", "sales", "profit", "loss_rate", "opportunity_score"]
            ].head(12),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Opportunity score blends sales, profit, loss rate, and discount discipline.")

    st.markdown("**State and Regional Breakdown**")
    state_summary = build_state_summary(filtered).head(20)
    fig = px.bar(
        state_summary,
        x="State",
        y="sales",
        color="Region",
        hover_data=["profit", "avg_discount"],
        title="Top States by Sales",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_customer_tab(filtered: pd.DataFrame):
    st.subheader("Customer Insights")
    st.caption("Customer segmentation helps identify high-value and at-risk customers.")

    customer_summary = build_customer_summary(filtered)
    rfm = build_rfm_summary(filtered)
    segment_counts = rfm.groupby("rfm_segment", as_index=False).agg(
        customers=("Customer ID", "count"),
        sales=("monetary", "sum"),
    )

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("**Top Customers**")
        st.dataframe(customer_summary.head(15), use_container_width=True, hide_index=True)

    with right:
        fig = px.bar(
            segment_counts,
            x="rfm_segment",
            y="customers",
            color="sales",
            title="RFM Customer Segmentation",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**High-Value Customer Watchlist**")
    watchlist = rfm[
        ["Customer Name", "recency_days", "frequency", "monetary", "profit", "rfm_segment"]
    ].head(20)
    st.dataframe(watchlist, use_container_width=True, hide_index=True)


def render_strategy_tab(filtered: pd.DataFrame):
    st.subheader("Business Planning and Forecasting")
    st.caption("Scenario analysis and basic forecasting for business planning.")

    adjustment = st.slider(
        "Adjust average discount by",
        min_value=-0.20,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )

    scenario = simulate_discount_change(filtered, adjustment)

    left, right = st.columns([1.05, 0.95])

    with left:
        fig = px.bar(
            scenario.head(12),
            x="Sub-Category",
            y="estimated_profit_impact",
            color="projected_profit",
            title="Estimated Profit Impact by Sub-Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.dataframe(
            scenario[
                [
                    "Sub-Category",
                    "avg_discount",
                    "simulated_discount",
                    "estimated_profit_impact",
                    "projected_profit",
                ]
            ].head(15),
            use_container_width=True,
            hide_index=True,
        )

    forecast = build_forecast_frame(filtered, periods=4)
    fig = px.line(
        forecast,
        x="order_month_ts",
        y=["sales", "profit"],
        color="type",
        line_dash="type",
        title="Historical Trend and Forecast",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ml_tab(filtered: pd.DataFrame, loss_result, profit_result):
    st.subheader("Predictive Models")
    st.markdown("### Predict Order Risk and Profit")
    st.caption(
        "This tool predicts whether an order may result in a loss and estimates expected profit."
    )

    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"**Loss Classifier Accuracy:** {loss_result.accuracy:.1%}")
        st.markdown(f"**Loss Classifier Precision:** {loss_result.precision:.1%}")
        st.markdown(f"**Loss Classifier Recall:** {loss_result.recall:.1%}")
        st.markdown(f"**Profit Regressor MAE:** {money(profit_result.mae)}")
        st.markdown(f"**Profit Regressor R²:** {profit_result.r2:.2f}")
        st.caption(f"Training rows: {loss_result.train_rows:,} | Test rows: {loss_result.test_rows:,}")

        sample = {
            "Category": st.selectbox("Category", sorted(filtered["Category"].dropna().unique())),
            "Sub-Category": st.selectbox(
                "Sub-Category", sorted(filtered["Sub-Category"].dropna().unique())
            ),
            "Region": st.selectbox("Region", sorted(filtered["Region"].dropna().unique())),
            "Segment": st.selectbox("Segment", sorted(filtered["Segment"].dropna().unique())),
            "Ship Mode": st.selectbox("Ship Mode", sorted(filtered["Ship Mode"].dropna().unique())),
            "Discount": st.slider("Discount", 0.0, 0.8, 0.2, 0.05),
            "Sales": st.number_input("Sales", min_value=1.0, value=250.0, step=25.0),
            "Quantity": st.slider("Quantity", 1, 14, 3),
            "shipping_days": st.slider("Shipping Days", 0, 8, 4),
        }

        sample_df = pd.DataFrame([sample])
        loss_probability = float(loss_result.pipeline.predict_proba(sample_df)[0][1])
        predicted_profit = float(profit_result.pipeline.predict(sample_df)[0])

        st.metric("Predicted Loss Risk", pct(loss_probability))
        st.progress(loss_probability)
        st.metric("Predicted Profit", money(predicted_profit))

        if loss_probability >= 0.6:
            st.warning("This order looks risky. Review discount, shipping time, or product mix.")
        elif predicted_profit < 0:
            st.warning("The model expects negative profit even though the risk score is not extremely high.")
        else:
            st.success("This sample order looks reasonably healthy based on the training data.")

    with right:
        risk_segments = build_discount_risk_segments(filtered)
        segment_df = pd.DataFrame(
            [
                {
                    "Discount Band": item.band_name,
                    "Average Sales": item.avg_sales,
                    "Average Profit": item.avg_profit,
                    "Loss Rate": item.loss_rate,
                    "Rows": item.rows,
                }
                for item in risk_segments
            ]
        )

        fig = go.Figure()
        fig.add_bar(
            x=segment_df["Discount Band"],
            y=segment_df["Average Profit"],
            name="Average Profit",
        )
        fig.add_scatter(
            x=segment_df["Discount Band"],
            y=segment_df["Loss Rate"],
            name="Loss Rate",
            yaxis="y2",
        )
        fig.update_layout(
            title="Impact of Discount Levels on Profit and Risk",
            yaxis=dict(title="Average Profit"),
            yaxis2=dict(title="Loss Rate", overlaying="y", side="right", tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

    eval_left, eval_right = st.columns(2)

    with eval_left:
        st.markdown("**Loss Model Confusion Matrix**")
        st.dataframe(get_confusion_matrix_frame(loss_result), use_container_width=True)

    with eval_right:
        st.markdown("**Profit Model Sample Prediction Errors**")
        st.dataframe(
            get_regression_preview_frame(profit_result),
            use_container_width=True,
            hide_index=True,
        )

    imp_left, imp_right = st.columns(2)

    with imp_left:
        loss_importance = get_feature_importance_table(loss_result.pipeline).head(12)
        fig = px.bar(
            loss_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Loss Model Features",
        )
        st.plotly_chart(fig, use_container_width=True)

    with imp_right:
        profit_importance = get_feature_importance_table(profit_result.pipeline).head(12)
        fig = px.bar(
            profit_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Profit Model Features",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_sql_tab():
    st.subheader("SQL-Based Analysis")
    st.caption("Query business data using SQL for flexible analysis.")

    db_file = get_sql_store()
    counts = get_table_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders in SQL", f"{counts['orders']:,}")
    c2.metric("Monthly KPI Rows", f"{counts['months']:,}")
    c3.metric("Category KPI Rows", f"{counts['categories']:,}")
    c4.metric("Customer KPI Rows", f"{counts['customers']:,}")


    selected_name = st.selectbox("Starter SQL Query", list(DEFAULT_SQL_QUERIES.keys()))
    default_query = DEFAULT_SQL_QUERIES[selected_name]
    query = st.text_area("SQL Editor", value=default_query, height=160)

    if st.button("Run SQL Query", type="primary"):
        try:
            result = run_sql_query(query)
            st.success(f"Query returned {len(result):,} rows.")
            st.dataframe(result, use_container_width=True, hide_index=True)

            if not result.empty and {"order_month", "sales", "profit"}.issubset(result.columns):
                fig = px.line(
                    result.sort_values("order_month"),
                    x="order_month",
                    y=["sales", "profit"],
                    title="SQL Query Result Preview",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    st.code(
        "select Region, round(sum(Sales), 2) as sales, round(sum(Profit), 2) as profit\n"
        "from orders\n"
        "group by Region\n"
        "order by profit desc;",
        language="sql",
    )


def render_live_monitor_tab(filtered: pd.DataFrame):
    st.subheader("Live Operations Monitor")
    st.caption("Displays recent orders and flags potential risks using recent activity.")

    if filtered.empty:
        st.warning("No rows are available for the current filter selection.")
        return

    date_col = get_date_column(filtered)
    if date_col is None:
        st.error("No date column found for the operations monitor.")
        st.write(list(filtered.columns))
        return

    working = filtered.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col])

    if working.empty:
        st.warning("No valid dated rows are available for the current filter selection.")
        return

    refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    recent_orders = working.sort_values(date_col, ascending=False).head(12).copy()
    recent_orders["risk_flag"] = recent_orders["Profit"].apply(
        lambda value: "High Risk" if value < 0 else "Normal"
    )
    recent_orders["sales_text"] = recent_orders["Sales"].map(money)
    recent_orders["profit_text"] = recent_orders["Profit"].map(money)

    alert_count = int((recent_orders["Profit"] < 0).sum())
    avg_shipping = float(recent_orders["shipping_days"].mean()) if len(recent_orders) else 0.0
    avg_discount = float(recent_orders["Discount"].mean()) if len(recent_orders) else 0.0

    left, right = st.columns([0.9, 1.1])

    with left:
        st.markdown(
            f"""
            <div class="status-card">
            <strong>Monitor Snapshot</strong><br>
            Last refresh: {refresh_time}<br>
            Recent orders reviewed: {len(recent_orders)}<br>
            Negative-profit orders in snapshot: {alert_count}<br>
            Average shipping days: {avg_shipping:.1f}<br>
            Average discount: {avg_discount:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Refresh Monitor"):
            st.rerun()

        anomalies = build_anomaly_table(working)
        alert_feed = anomalies[anomalies["alert_flag"]][
            ["order_month", "sales", "profit", "alert_reason"]
        ].head(6)

        st.markdown("**Recent Alert Feed**")
        if alert_feed.empty:
            st.success("No active monthly alerts in the current view.")
        else:
            st.dataframe(alert_feed, use_container_width=True, hide_index=True)

    with right:
        st.markdown("**Recent Orders Snapshot**")
        display_columns = [
            date_col,
            "Order ID",
            "Customer Name",
            "Category",
            "Sub-Category",
            "Region",
            "sales_text",
            "profit_text",
            "risk_flag",
        ]
        st.dataframe(
            recent_orders[display_columns],
            use_container_width=True,
            hide_index=True,
        )

    day_summary = (
        working.assign(order_day=working[date_col].dt.date)
        .groupby("order_day", as_index=False)
        .agg(
            sales=("Sales", "sum"),
            profit=("Profit", "sum"),
            orders=("Order ID", "nunique"),
        )
        .sort_values("order_day", ascending=False)
        .head(14)
        .sort_values("order_day")
    )

    fig = px.bar(
        day_summary,
        x="order_day",
        y="orders",
        color="profit",
        hover_data=["sales"],
        title="Recent Daily Order Pulse",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_quality_tab(filtered: pd.DataFrame):
    st.subheader("Data Quality and Validation")
    st.caption("Quick checks for completeness, consistency, and export readiness.")

    quality = build_data_quality_summary(filtered)
    st.dataframe(quality, use_container_width=True, hide_index=True)

    recommendations = pd.DataFrame(
        {
            "Focus Area": [
                "Discount policy",
                "Low-profit sub-categories",
                "High-value customers",
                "Regional planning",
                "Operational monitoring",
            ],
            "Why It Matters": [
                "Higher discounts are often tied to weaker average profit.",
                "Some products sell well but still drag down margin.",
                "RFM segments help identify retention priorities.",
                "Regional views highlight where sales are high but profit may lag.",
                "Anomaly checks help catch unusual months quickly.",
            ],
            "Action Idea": [
                "Review large discounts before approval.",
                "Check pricing, bundle strategy, and shipping costs.",
                "Build targeted retention offers for champions and loyal customers.",
                "Compare state performance before expanding promotions.",
                "Escalate sudden profit dips for root-cause analysis.",
            ],
        }
    )
    st.dataframe(recommendations, use_container_width=True, hide_index=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data as CSV",
        csv_data,
        file_name="filtered_superstore_view.csv",
        mime="text/csv",
    )


def main():
    df = get_data()
    get_sql_store()
    loss_result, profit_result = get_models()

    render_header(len(df))

    with st.sidebar:
        st.header("Filters")

        years = st.multiselect(
            "Order Year",
            sorted(df["order_year"].unique()),
            default=sorted(df["order_year"].unique()),
        )
        regions = st.multiselect(
            "Region",
            sorted(df["Region"].dropna().unique()),
            default=sorted(df["Region"].dropna().unique()),
        )
        categories = st.multiselect(
            "Category",
            sorted(df["Category"].dropna().unique()),
            default=sorted(df["Category"].dropna().unique()),
        )
        segments = st.multiselect(
            "Segment",
            sorted(df["Segment"].dropna().unique()),
            default=sorted(df["Segment"].dropna().unique()),
        )
        states = st.multiselect(
            "State",
            sorted(df["State"].dropna().unique()),
            default=sorted(df["State"].dropna().unique()),
        )
        ship_modes = st.multiselect(
            "Ship Mode",
            sorted(df["Ship Mode"].dropna().unique()),
            default=sorted(df["Ship Mode"].dropna().unique()),
        )

        if st.button("Reset Filters"):
            st.rerun()

    filtered = apply_filters(df, years, regions, categories, segments, states, ship_modes)

    if filtered.empty:
        st.error("No rows match the current filter selection. Please widen the filters.")
        return

    render_kpis(filtered)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Executive Dashboard",
            "Sales Performance",
            "Customer Insights",
            "Forecasting & Strategy",
            "Predictive Models",
            "SQL Analysis",
            "Operations Monitor",
            "Data Quality",
        ]
    )

    with tab1:
        render_executive_tab(filtered)
    with tab2:
        render_commercial_tab(filtered)
    with tab3:
        render_customer_tab(filtered)
    with tab4:
        render_strategy_tab(filtered)
    with tab5:
        render_ml_tab(filtered, loss_result, profit_result)
    with tab6:
        render_sql_tab()
    with tab7:
        render_live_monitor_tab(filtered)
    with tab8:
        render_quality_tab(filtered)


if __name__ == "__main__":
    main()