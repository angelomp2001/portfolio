from pathlib import Path

# Directory containing this config.py file: .../project_2/src
CONFIG_DIR = Path(__file__).resolve().parent

# Project root: .../project_2
PROJECT_ROOT = CONFIG_DIR.parent

# Data directory and individual files (absolute Paths)
DATA_DIR = PROJECT_ROOT / "data"

ORDERS_PATH = DATA_DIR / "instacart_orders.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
DEPARTMENTS_PATH = DATA_DIR / "departments.csv"
AISLES_PATH = DATA_DIR / "aisles.csv"
ORDER_PRODUCTS_PATH = DATA_DIR / "order_products.csv"