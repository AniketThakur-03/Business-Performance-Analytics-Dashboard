# Business Performance Analytics Dashboard

This project analyzes business performance using the Superstore dataset. It focuses on sales, profit, customer behavior, and order-level trends through an interactive dashboard.

---

## Overview

The goal of this project was to build a practical analytics dashboard that combines data analysis, SQL, and basic machine learning in one application.

It allows users to explore business data, apply filters, run SQL queries, and test simple predictive models to better understand profit patterns and order risk.

---

## Key Features

### Executive Dashboard
- Displays key metrics such as revenue, profit, margin, orders, and customers
- Includes month-over-month comparisons for quick performance tracking

### Sales Performance
- Analyzes category and sub-category performance
- Highlights the relationship between discounts and profitability

### Customer Insights
- Identifies top customers by sales
- Uses simple RFM segmentation to group customer value

### Forecasting and Strategy
- Shows sales and profit trends over time
- Includes a discount scenario tool for decision support

### Predictive Models
- Estimates loss risk for orders
- Predicts expected profit using a baseline regression model

### SQL Analysis
- Allows users to run SQL queries on the dataset

### Operations Monitor
- Displays recent orders
- Flags potentially risky orders for review

### Data Quality
- Includes basic validation checks
- Supports exporting filtered data

---

## Business Insights

- Higher discounts are strongly linked to lower profitability.
- Orders with discounts above 20% show a noticeable decline in profit.
- Very high discounts, especially above 40%, often lead to negative profit.
- A small group of customers contributes a large share of total revenue.
- Some categories generate strong sales but weaker profit margins.

These findings show that discount control and product-level strategy are important for improving business performance.

---

## Model Results

### Loss Risk Model (Classification)
- Accuracy: 92.8%
- Precision: 76.3%
- Recall: 89.6%

This model performs well in identifying risky orders. The high recall is useful because it helps capture most loss-making orders before they have a larger business impact.

### Profit Prediction Model (Regression)
- Mean Absolute Error (MAE): $27
- R² Score: 0.04

This is a baseline regression model. It captures general patterns, but it does not explain profit variation strongly because profit is affected by many business factors, including discounting, category mix, and customer behavior.

### Key Observations
- Discount is a major driver of both profit and loss risk.
- Higher discount ranges significantly increase the probability of loss.
- The classification model can help flag risky orders before execution.

---

## Tech Stack

- Python
- Pandas
- Streamlit
- Plotly
- SQLite
- Scikit-learn

---
  
## Screenshots

### Dashboard
![Dashboard](Assets/Dashboard.png)


### Predictive Models
![Predictive Models](Assets/Predictive.png)

### SQL Analysis
![SQL](Assets/SQL.png)

### Operations Monitor
![Operations](Assets/Operations.png)

---
## Future Improvements

- Improve regression model performance with feature engineering
- Add real-time data support
- Deploy dashboard for live access

---
```md
## Getting Started
python -m venv .venv
pip install -r requirements.txt
.venv\Scripts\Activate.ps1
python scripts/build_outputs.py
streamlit run app.py
