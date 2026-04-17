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
- Uses simple RFM segmentation to group customers  

Forecasting and Strategy  
- Displays sales and profit trends  
- Includes a basic discount scenario tool  

Predictive Models  
- Estimates whether an order may result in a loss  
- Predicts expected profit based on input values  

SQL Analysis  
- Allows running SQL queries on the dataset  
- Includes a few predefined queries for quick analysis  

Operations Monitor  
- Shows recent orders  
- Flags potentially risky orders  

Data Quality  
- Displays basic data checks  
- Allows exporting filtered data  

---

## Tech Stack

- Python  
- Pandas  
- Streamlit  
- Plotly  
- SQLite  
- Scikit-learn  
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
