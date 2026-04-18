# Business Performance Analytics Dashboard

This project analyzes business performance using the Superstore dataset. It focuses on understanding sales, profit, customer behavior, and order-level trends through an interactive dashboard.

---

## Overview

The goal of this project was to build a practical analytics dashboard that brings together data analysis, SQL, and basic machine learning in one place.

It allows users to explore business data, filter results, run SQL queries, and test simple predictive models to better understand profit and risk patterns.

---

## Key Features

Executive Dashboard  
- Shows key metrics like revenue, profit, margin, orders, and customers  
- Includes month-over-month comparison  

Sales Performance  
- Analyzes category and sub-category performance  
- Highlights how discounts affect profitability  

Customer Insights  
- Identifies top customers  
- Uses simple RFM segmentation  

Forecasting and Strategy  
- Displays sales and profit trends  
- Includes a discount scenario tool  

Predictive Models  
- Estimates loss risk  
- Predicts expected profit  

SQL Analysis  
- Run SQL queries on the dataset  

Operations Monitor  
- Shows recent orders  
- Flags risky orders  

Data Quality  
- Basic validation checks  
- Export filtered data  

---

## Business Insights

- Higher discounts are strongly linked to lower profitability.  
- Orders with discounts above 20% show a sharp decline in profit.  
- Very high discounts (above 40%) often result in negative profit.  
- A small group of customers contributes a large portion of revenue.  
- Some categories generate strong sales but weak margins.  

These insights highlight the importance of discount control and product-level strategy.

---

## Model Results

Loss Risk Model (Classification)
- Accuracy: 92.8%
- Precision: 76.3%
- Recall: 89.6%

The model performs well at identifying risky orders. High recall ensures most loss-making orders are captured.

Profit Prediction Model (Regression)
- Mean Absolute Error (MAE): $27  
- R² Score: 0.04  

This is a baseline model. It captures general trends but does not fully explain profit variation due to multiple influencing factors.

### Key Observations

- Discount is a major driver of both profit and loss risk.  
- Higher discount ranges significantly increase loss probability.  
- The model can help flag risky orders before execution.  

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
```md
## Getting Started
python -m venv .venv
pip install -r requirements.txt
.venv\Scripts\Activate.ps1
python scripts/build_outputs.py
streamlit run app.py
