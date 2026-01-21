'''
Provide insights into the shopping habits of Instacart customers
'''
from src.data_preprocessing import load_data, fill_missing, remove_duplicates, verify_data
from src.analysis import (
    difference_in_order_hours,
    plot_orders_per_customer,
    plot_top_popular_products,
    plot_items_per_order,
    plot_top_reordered_items,
    plot_top_first_items
)

def main():
    # data path
    orders_path = 'data/instacart_orders.csv'
    products_path = 'data/products.csv'
    departments_path = 'data/departments.csv'
    aisles_path = 'data/aisles.csv'
    order_products_path = 'data/order_products.csv'

    # load data
    print("Loading data...")
    orders = load_data(orders_path)
    products = load_data(products_path)
    departments = load_data(departments_path)
    aisles = load_data(aisles_path)
    order_products = load_data(order_products_path)

    # fill missing values
    print("Handling missing values...")
    products['product_name'] = fill_missing(products['product_name'])
    order_products['add_to_cart_order'] = fill_missing(order_products['add_to_cart_order'])

    # remove duplicates
    print("Removing duplicates...")
    orders = remove_duplicates(orders)
    products = remove_duplicates(products)
    departments = remove_duplicates(departments)
    aisles = remove_duplicates(aisles)
    order_products = remove_duplicates(order_products)

    # verify that col values make sense
    print("Verifying data...")
    verify_data(orders)

    # 1. Difference in 'order_hour_of_day' distributions on Wednesdays and Saturdays
    difference_in_order_hours(orders, 'order_dow', [3, 6])

    # 2. Distribution of number of orders per customer
    plot_orders_per_customer(orders)

    # 3. Top 20 popular products
    plot_top_popular_products(order_products, products)

    # 4. Number of items bought per order
    plot_items_per_order(order_products)

    # 5. Top 20 reordered items
    plot_top_reordered_items(order_products, products)
    
    # 6. Top 20 'first' items in cart
    plot_top_first_items(order_products, products)

if __name__ == '__main__':
    main()