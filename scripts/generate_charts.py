from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Superstore.csv"
OUTPUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def money(value):
    return f"${value:,.2f}"


def train_model(df: pd.DataFrame):
    features = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode", "Discount", "Sales", "Quantity", "shipping_days"]
    X = df[features]
    y = (df["Profit"] < 0).astype(int)
    categorical = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
    numeric = ["Discount", "Sales", "Quantity", "shipping_days"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ])),
        ("model", RandomForestClassifier(n_estimators=220, max_depth=10, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return accuracy_score(y_test, preds), classification_report(y_test, preds, output_dict=True)


df = pd.read_csv(DATA_PATH, encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["order_year"] = df["Order Date"].dt.year
df["order_month"] = df["Order Date"].dt.to_period("M").astype(str)
df["shipping_days"] = (df["Ship Date"] - df["Order Date"]).dt.days

def build_standard_outputs(df: pd.DataFrame):
    sales_by_year = df.groupby("order_year")["Sales"].sum()
    fig = plt.figure()
    sales_by_year.plot(marker="o")
    plt.title("Total Sales by Year")
    plt.xlabel("Year")
    plt.ylabel("Sales")
    save_plot(fig, "sales_by_year.png")

    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    fig = plt.figure()
    category_sales.plot(kind="bar")
    plt.title("Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Sales")
    save_plot(fig, "sales_by_category.png")

    subcat_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()
    fig = plt.figure(figsize=(8, 6))
    subcat_profit.plot(kind="barh")
    plt.title("Profit by Sub-Category")
    plt.xlabel("Profit")
    plt.ylabel("Sub-Category")
    save_plot(fig, "profit_by_subcategory.png")

    discount_profit = df.groupby("Discount")["Profit"].mean().sort_index()
    fig = plt.figure()
    discount_profit.plot(marker="o")
    plt.title("Average Profit by Discount Level")
    plt.xlabel("Discount")
    plt.ylabel("Average Profit")
    save_plot(fig, "profit_by_discount.png")

    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    fig = plt.figure()
    region_sales.plot(kind="bar")
    plt.title("Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Sales")
    save_plot(fig, "sales_by_region.png")

    monthly_sales = df.groupby("order_month")["Sales"].sum().tail(18)
    fig = plt.figure(figsize=(10, 4))
    monthly_sales.plot(marker="o")
    plt.title("Monthly Sales Trend (Last 18 Months in Dataset)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    save_plot(fig, "monthly_sales_trend.png")

    kpi_summary = pd.DataFrame([
        {"metric": "Total Sales", "value": round(df["Sales"].sum(), 2)},
        {"metric": "Total Profit", "value": round(df["Profit"].sum(), 2)},
        {"metric": "Profit Margin", "value": round(df["Profit"].sum() / df["Sales"].sum(), 4)},
        {"metric": "Average Discount", "value": round(df["Discount"].mean(), 4)},
        {"metric": "Average Shipping Days", "value": round(df["shipping_days"].mean(), 2)},
        {"metric": "Orders", "value": int(df["Order ID"].nunique())},
        {"metric": "Customers", "value": int(df["Customer ID"].nunique())},
    ])
    kpi_summary.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)

    top_customers = (
        df.groupby(["Customer ID", "Customer Name"], as_index=False)
        .agg(total_sales=("Sales", "sum"), total_profit=("Profit", "sum"), orders=("Order ID", "nunique"))
        .sort_values("total_sales", ascending=False)
        .head(10)
    )
    top_customers.to_csv(OUTPUT_DIR / "top_customers.csv", index=False)

    accuracy, report = train_model(df)
    insights = f"""# Key Insights

## KPI Summary
- Total sales: {money(df['Sales'].sum())}
- Total profit: {money(df['Profit'].sum())}
- Profit margin: {df['Profit'].sum() / df['Sales'].sum():.2%}
- Average discount: {df['Discount'].mean():.2%}
- Average shipping time: {df['shipping_days'].mean():.2f} days
- Loss-making rows: {(df['Profit'] < 0).mean():.2%}

## Highlights
- Highest-sales category: {category_sales.idxmax()} ({money(category_sales.max())})
- Best region by sales: {region_sales.idxmax()} ({money(region_sales.max())})
- Strongest sub-category by profit: {subcat_profit.idxmax()} ({money(subcat_profit.max())})
- Weakest sub-category by profit: {subcat_profit.idxmin()} ({money(subcat_profit.min())})
- Discount level with worst average profit: {discount_profit.idxmin()} ({money(discount_profit.min())})

## Predictive layer
- Loss-order model accuracy: {accuracy:.2%}
- Precision for profitable class: {report['0']['precision']:.2%}
- Precision for loss-making class: {report['1']['precision']:.2%}

## Recommended storyline
1. Sales grew over time, but profit is uneven across categories and discount levels.
2. Discount strategy deserves attention because higher discounts reduce average profit.
3. Loss-making sub-categories should be reviewed for pricing, returns, and fulfillment costs.
4. The predictive section flags risky orders before they become margin problems.
"""
    (DOCS_DIR / "key_insights.md").write_text(insights)

build_standard_outputs(df)
print("Charts, CSV summaries, ML-enhanced insights, and docs generated.")
