from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'raw' / 'Superstore.csv'


def load_superstore_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding='latin1').copy()

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['order_year'] = df['Order Date'].dt.year
    df['order_month'] = df['Order Date'].dt.to_period('M').astype(str)
    df['order_month_ts'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    df['order_quarter'] = df['Order Date'].dt.to_period('Q').astype(str)
    df['shipping_days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['profit_margin'] = (df['Profit'] / df['Sales'].replace(0, pd.NA)).fillna(0.0)
    df['is_loss'] = (df['Profit'] < 0).astype(int)
    df['discount_band'] = pd.cut(
        df['Discount'],
        bins=[-0.01, 0.0, 0.1, 0.2, 0.4, 0.8],
        labels=['0%', '1-10%', '11-20%', '21-40%', '41%+'],
    )
    df['sales_band'] = pd.qcut(
        df['Sales'].rank(method='first'),
        q=4,
        labels=['Low', 'Medium', 'High', 'Very High'],
    )
    df['days_from_latest_order'] = (df['Order Date'].max() - df['Order Date']).dt.days
    return df



def apply_filters(
    df: pd.DataFrame,
    years: list[int],
    regions: list[str],
    categories: list[str],
    segments: list[str],
    states: list[str] | None = None,
    ship_modes: list[str] | None = None,
) -> pd.DataFrame:
    filtered = df[
        df['order_year'].isin(years)
        & df['Region'].isin(regions)
        & df['Category'].isin(categories)
        & df['Segment'].isin(segments)
    ].copy()

    if states is not None and len(states) > 0:
        filtered = filtered[filtered['State'].isin(states)].copy()
    if ship_modes is not None and len(ship_modes) > 0:
        filtered = filtered[filtered['Ship Mode'].isin(ship_modes)].copy()

    return filtered



def build_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    customer_summary = (
        df.groupby(['Customer ID', 'Customer Name', 'Segment'], as_index=False)
        .agg(
            sales=('Sales', 'sum'),
            profit=('Profit', 'sum'),
            orders=('Order ID', 'nunique'),
            avg_discount=('Discount', 'mean'),
            last_order=('Order Date', 'max'),
        )
        .sort_values(['sales', 'profit'], ascending=[False, False])
    )
    return customer_summary



def build_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    state_summary = (
        df.groupby(['State', 'Region'], as_index=False)
        .agg(
            sales=('Sales', 'sum'),
            profit=('Profit', 'sum'),
            orders=('Order ID', 'nunique'),
            avg_discount=('Discount', 'mean'),
        )
        .sort_values('sales', ascending=False)
    )
    return state_summary



def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(['order_month', 'order_month_ts'], as_index=False)
        .agg(
            sales=('Sales', 'sum'),
            profit=('Profit', 'sum'),
            orders=('Order ID', 'nunique'),
        )
        .sort_values('order_month_ts')
    )
    monthly['sales_growth_pct'] = monthly['sales'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    monthly['profit_growth_pct'] = monthly['profit'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return monthly



def build_executive_kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    monthly = build_monthly_summary(df)
    current = monthly.iloc[-1] if len(monthly) else None
    previous = monthly.iloc[-2] if len(monthly) > 1 else None

    total_sales = float(df['Sales'].sum())
    total_profit = float(df['Profit'].sum())
    total_orders = float(df['Order ID'].nunique())
    total_customers = float(df['Customer ID'].nunique())
    profit_margin = float(total_profit / total_sales) if total_sales else 0.0
    loss_rate = float(df['is_loss'].mean()) if len(df) else 0.0

    sales_delta = 0.0
    profit_delta = 0.0
    if current is not None and previous is not None:
        prev_sales = float(previous['sales'])
        prev_profit = float(previous['profit'])
        sales_delta = float((current['sales'] - prev_sales) / prev_sales) if prev_sales else 0.0
        profit_delta = float((current['profit'] - prev_profit) / prev_profit) if prev_profit else 0.0

    return {
        'total_sales': total_sales,
        'total_profit': total_profit,
        'total_orders': total_orders,
        'total_customers': total_customers,
        'profit_margin': profit_margin,
        'loss_rate': loss_rate,
        'sales_delta': sales_delta,
        'profit_delta': profit_delta,
    }



def build_rfm_summary(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df['Order Date'].max()
    rfm = (
        df.groupby(['Customer ID', 'Customer Name'], as_index=False)
        .agg(
            recency_days=('Order Date', lambda x: (latest_date - x.max()).days),
            frequency=('Order ID', 'nunique'),
            monetary=('Sales', 'sum'),
            profit=('Profit', 'sum'),
        )
    )

    rfm['recency_score'] = pd.qcut(rfm['recency_days'].rank(method='first', ascending=False), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['monetary_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['rfm_score'] = rfm['recency_score'] + rfm['frequency_score'] + rfm['monetary_score']

    def label_customer(score: int) -> str:
        if score >= 10:
            return 'Champion'
        if score >= 8:
            return 'Loyal'
        if score >= 6:
            return 'Promising'
        return 'At Risk'

    rfm['rfm_segment'] = rfm['rfm_score'].apply(label_customer)
    return rfm.sort_values(['rfm_score', 'monetary'], ascending=[False, False])



def build_forecast_frame(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    monthly = build_monthly_summary(df).copy()
    monthly = monthly[['order_month_ts', 'sales', 'profit']].reset_index(drop=True)
    monthly['month_index'] = range(len(monthly))

    if len(monthly) < 2:
        monthly['type'] = 'historical'
        return monthly

    sales_coef = np.polyfit(monthly['month_index'], monthly['sales'], deg=1)
    profit_coef = np.polyfit(monthly['month_index'], monthly['profit'], deg=1)

    future_rows = []
    last_ts = monthly['order_month_ts'].max()
    start_idx = int(monthly['month_index'].max()) + 1
    for step in range(periods):
        idx = start_idx + step
        future_rows.append(
            {
                'order_month_ts': last_ts + pd.DateOffset(months=step + 1),
                'sales': max(0.0, float(np.polyval(sales_coef, idx))),
                'profit': float(np.polyval(profit_coef, idx)),
                'month_index': idx,
                'type': 'forecast',
            }
        )

    monthly['type'] = 'historical'
    return pd.concat([monthly, pd.DataFrame(future_rows)], ignore_index=True)



def build_product_opportunity_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(['Category', 'Sub-Category'], as_index=False)
        .agg(
            sales=('Sales', 'sum'),
            profit=('Profit', 'sum'),
            orders=('Order ID', 'nunique'),
            avg_discount=('Discount', 'mean'),
            loss_rate=('is_loss', 'mean'),
        )
    )
    summary['profit_margin'] = summary['profit'] / summary['sales'].replace(0, np.nan)
    summary['opportunity_score'] = (
        summary['sales'].rank(pct=True) * 0.35
        + (1 - summary['loss_rate']).rank(pct=True) * 0.25
        + summary['profit'].rank(pct=True) * 0.25
        + (1 - summary['avg_discount']).rank(pct=True) * 0.15
    )
    return summary.sort_values('opportunity_score', ascending=False)



def build_anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    monthly = build_monthly_summary(df).copy()
    sales_mean = monthly['sales'].mean()
    sales_std = monthly['sales'].std(ddof=0) or 1.0
    profit_mean = monthly['profit'].mean()
    profit_std = monthly['profit'].std(ddof=0) or 1.0

    monthly['sales_zscore'] = (monthly['sales'] - sales_mean) / sales_std
    monthly['profit_zscore'] = (monthly['profit'] - profit_mean) / profit_std
    monthly['alert_flag'] = (monthly['sales_zscore'].abs() >= 1.5) | (monthly['profit_zscore'].abs() >= 1.5)
    monthly['alert_reason'] = np.where(
        monthly['alert_flag'],
        np.where(monthly['profit_zscore'] < -1.5, 'Profit dip', 'Demand spike / unusual movement'),
        'Normal',
    )
    return monthly.sort_values('order_month_ts', ascending=False)



def build_data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in ['Sales', 'Profit', 'Discount', 'Quantity', 'shipping_days', 'Region', 'State', 'Category', 'Sub-Category']:
        records.append(
            {
                'column': col,
                'missing_values': int(df[col].isna().sum()),
                'unique_values': int(df[col].nunique(dropna=True)),
                'sample_value': str(df[col].dropna().iloc[0]) if df[col].notna().any() else 'N/A',
            }
        )
    return pd.DataFrame(records)



def simulate_discount_change(df: pd.DataFrame, adjustment: float) -> pd.DataFrame:
    summary = (
        df.groupby('Sub-Category', as_index=False)
        .agg(
            sales=('Sales', 'sum'),
            profit=('Profit', 'sum'),
            avg_discount=('Discount', 'mean'),
        )
    )
    summary['simulated_discount'] = (summary['avg_discount'] + adjustment).clip(0, 0.8)
    summary['discount_change'] = summary['simulated_discount'] - summary['avg_discount']
    summary['estimated_profit_impact'] = summary['sales'] * summary['discount_change'] * -0.35
    summary['projected_profit'] = summary['profit'] + summary['estimated_profit_impact']
    return summary.sort_values('estimated_profit_impact')
