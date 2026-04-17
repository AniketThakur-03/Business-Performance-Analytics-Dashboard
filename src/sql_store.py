from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'data' / 'sql' / 'superstore_analytics.db'


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    target = db_path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    return conn


def build_sqlite_store(df: pd.DataFrame, db_path: Path | None = None) -> Path:
    target = db_path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    for col in ['Order Date', 'Ship Date', 'order_month_ts']:
        if col in export_df.columns:
            export_df[col] = export_df[col].astype(str)

    with get_connection(target) as conn:
        export_df.to_sql('orders', conn, if_exists='replace', index=False)
        conn.execute('drop view if exists monthly_kpis')
        conn.execute(
            '''
            create view monthly_kpis as
            select
                order_month,
                round(sum(Sales), 2) as sales,
                round(sum(Profit), 2) as profit,
                count(distinct "Order ID") as orders,
                count(distinct "Customer ID") as customers,
                round(avg(Discount), 4) as avg_discount,
                round(avg(shipping_days), 2) as avg_shipping_days,
                round(avg(is_loss), 4) as loss_rate
            from orders
            group by order_month
            order by order_month
            '''
        )
        conn.execute('drop view if exists category_kpis')
        conn.execute(
            '''
            create view category_kpis as
            select
                Category,
                "Sub-Category" as sub_category,
                round(sum(Sales), 2) as sales,
                round(sum(Profit), 2) as profit,
                count(distinct "Order ID") as orders,
                round(avg(Discount), 4) as avg_discount,
                round(avg(is_loss), 4) as loss_rate
            from orders
            group by Category, "Sub-Category"
            order by sales desc
            '''
        )
        conn.execute('drop view if exists customer_kpis')
        conn.execute(
            '''
            create view customer_kpis as
            select
                "Customer ID" as customer_id,
                "Customer Name" as customer_name,
                Segment,
                round(sum(Sales), 2) as sales,
                round(sum(Profit), 2) as profit,
                count(distinct "Order ID") as orders,
                round(avg(Discount), 4) as avg_discount,
                max("Order Date") as last_order_date
            from orders
            group by "Customer ID", "Customer Name", Segment
            order by sales desc
            '''
        )
        conn.execute('create index if not exists idx_orders_month on orders(order_month)')
        conn.execute('create index if not exists idx_orders_region on orders(Region)')
        conn.execute('create index if not exists idx_orders_category on orders(Category)')
        conn.execute('create index if not exists idx_orders_customer on orders("Customer ID")')
    return target


DEFAULT_SQL_QUERIES: dict[str, str] = {
    'Monthly KPI trend': (
        'select order_month, sales, profit, orders, customers, avg_discount, loss_rate '\
        'from monthly_kpis order by order_month desc limit 18'
    ),
    'Most profitable sub-categories': (
        'select Category, sub_category, sales, profit, orders, avg_discount, loss_rate '\
        'from category_kpis order by profit desc limit 15'
    ),
    'Highest loss-rate sub-categories': (
        'select Category, sub_category, sales, profit, avg_discount, loss_rate '\
        'from category_kpis where orders >= 20 order by loss_rate desc, sales desc limit 15'
    ),
    'Best customers by profit': (
        'select customer_name, Segment, sales, profit, orders, avg_discount, last_order_date '\
        'from customer_kpis order by profit desc limit 15'
    ),
}


def run_sql_query(query: str, db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)



def get_table_counts(db_path: Path | None = None) -> dict[str, int]:
    with get_connection(db_path) as conn:
        orders = conn.execute('select count(*) from orders').fetchone()[0]
        months = conn.execute('select count(*) from monthly_kpis').fetchone()[0]
        categories = conn.execute('select count(*) from category_kpis').fetchone()[0]
        customers = conn.execute('select count(*) from customer_kpis').fetchone()[0]
    return {
        'orders': int(orders),
        'months': int(months),
        'categories': int(categories),
        'customers': int(customers),
    }
