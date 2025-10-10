'''
Provide insights into the shopping habits of Instacart customers
'''

from src.data_preprocessing import load_data, fill_missing, remove_duplicates, verify_data

# data path
orders_path = 'data/instacart_orders.csv'
products_path = 'data/products.csv'
departments_path = 'data/departments.csv'
aisles_path = 'data/aisles.csv'
order_products_path = 'data/order_products.csv'

# load data
orders = load_data(orders_path)
products = load_data(products_path)
departments = load_data(departments_path)
aisles = load_data(aisles_path)
order_products = load_data(order_products_path)

# fill missing values
fill_missing(products['product_name'])
fill_missing(order_products['add_to_cart_order'])

# remove duplicates
remove_duplicates(orders)
remove_duplicates(products)
remove_duplicates(departments)
remove_duplicates(aisles)
remove_duplicates(order_products)


# verify that col values make sense
verify_data(orders)
