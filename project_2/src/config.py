from pathlib import Path

# path to project root (assuming this file is in src/)
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / 'data'

# data path
orders_path = data_dir / 'instacart_orders.csv'
products_path = data_dir / 'products.csv'
departments_path = data_dir / 'departments.csv'
aisles_path = data_dir / 'aisles.csv'
order_products_path = data_dir / 'order_products.csv'