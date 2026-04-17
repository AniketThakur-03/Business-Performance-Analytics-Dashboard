from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data_utils import build_customer_summary, build_monthly_summary, build_state_summary, load_superstore_data
from src.modeling import build_discount_risk_segments, get_confusion_matrix_frame, train_loss_classifier, train_profit_regressor
from src.sql_store import build_sqlite_store, get_table_counts

OUTPUT_DIR = ROOT / 'outputs'
DOCS_DIR = ROOT / 'docs'
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


def save_plot(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)



def money(value: float) -> str:
    return f'${value:,.2f}'



def main() -> None:
    df = load_superstore_data()
    monthly = build_monthly_summary(df)
    customer_summary = build_customer_summary(df)
    state_summary = build_state_summary(df)
    loss_result = train_loss_classifier(df)
    profit_result = train_profit_regressor(df)
    risk_segments = build_discount_risk_segments(df)
    build_sqlite_store(df)
    sql_counts = get_table_counts()

    fig = plt.figure(figsize=(9, 4))
    plt.plot(monthly['order_month'], monthly['sales'], marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.title('Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    save_plot(fig, 'monthly_sales.png')

    category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
    fig = plt.figure()
    category_profit.plot(kind='bar')
    plt.title('Profit by Category')
    plt.xlabel('Category')
    plt.ylabel('Profit')
    save_plot(fig, 'profit_by_category.png')

    subcategory_profit = df.groupby('Sub-Category')['Profit'].sum().sort_values()
    fig = plt.figure(figsize=(8, 6))
    subcategory_profit.plot(kind='barh')
    plt.title('Profit by Sub-Category')
    plt.xlabel('Profit')
    plt.ylabel('Sub-Category')
    save_plot(fig, 'profit_by_subcategory.png')

    region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    fig = plt.figure()
    region_sales.plot(kind='bar')
    plt.title('Sales by Region')
    plt.xlabel('Region')
    plt.ylabel('Sales')
    save_plot(fig, 'sales_by_region.png')

    risk_df = pd.DataFrame(
        {
            'discount_band': [item.band_name for item in risk_segments],
            'avg_profit': [item.avg_profit for item in risk_segments],
            'loss_rate': [item.loss_rate for item in risk_segments],
        }
    )
    fig = plt.figure()
    plt.plot(risk_df['discount_band'], risk_df['avg_profit'], marker='o')
    plt.title('Average Profit by Discount Band')
    plt.xlabel('Discount Band')
    plt.ylabel('Average Profit')
    save_plot(fig, 'profit_by_discount_band.png')

    kpis = pd.DataFrame(
        [
            {'metric': 'Total sales', 'value': round(df['Sales'].sum(), 2)},
            {'metric': 'Total profit', 'value': round(df['Profit'].sum(), 2)},
            {'metric': 'Profit margin', 'value': round(df['Profit'].sum() / df['Sales'].sum(), 4)},
            {'metric': 'Orders', 'value': int(df['Order ID'].nunique())},
            {'metric': 'Customers', 'value': int(df['Customer ID'].nunique())},
            {'metric': 'Average shipping days', 'value': round(df['shipping_days'].mean(), 2)},
            {'metric': 'Loss row rate', 'value': round(df['is_loss'].mean(), 4)},
            {'metric': 'Loss model accuracy', 'value': round(loss_result.accuracy, 4)},
            {'metric': 'Profit model R2', 'value': round(profit_result.r2, 4)},
            {'metric': 'Loss model precision', 'value': round(loss_result.precision, 4)},
            {'metric': 'Loss model recall', 'value': round(loss_result.recall, 4)},
        ]
    )
    kpis.to_csv(OUTPUT_DIR / 'kpi_summary.csv', index=False)
    customer_summary.head(20).to_csv(OUTPUT_DIR / 'top_customers.csv', index=False)
    state_summary.head(20).to_csv(OUTPUT_DIR / 'top_states.csv', index=False)

    notes = f'''# Project notes

## Dataset snapshot
- Rows: {len(df):,}
- Order years: {df['order_year'].min()} to {df['order_year'].max()}
- Total sales: {money(df['Sales'].sum())}
- Total profit: {money(df['Profit'].sum())}
- Profit margin: {df['Profit'].sum() / df['Sales'].sum():.2%}

## What this project now includes
- interactive Streamlit dashboard with cleaner tabs
- customer and state level drill-down tables
- loss-risk classifier for negative-profit orders
- profit regressor for expected order profit
- export-ready PNG charts, CSV summaries, and a local SQLite analytics store

## Practical takeaways
1. Discounting needs closer control because high discounts are linked to weaker average profit.
2. A few sub-categories drag down profit even when they keep sales moving.
3. The predictive layer helps explain how a student can move from reporting into entry-level business analytics or junior data science work.
4. SQL views now support repeatable business queries instead of relying only on notebook logic.
5. The project is strong enough for screenshots, GitHub, and resume bullet points.
'''
    (DOCS_DIR / 'project_notes.md').write_text(notes, encoding='utf-8')
    print('Outputs refreshed successfully.')


if __name__ == '__main__':
    main()
