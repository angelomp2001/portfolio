import pandas as pd
import matplotlib.pyplot as plt

def difference_in_order_hours(df, column, days):
    print("\nAnalyzing Wednesday vs Saturday order hours...")
    wed_orders = df[df[column] == days[0]]['order_hour_of_day'].value_counts().sort_index()
    sat_orders = df[df[column] == days[1]]['order_hour_of_day'].value_counts().sort_index()

    compare_days = pd.concat([wed_orders, sat_orders], axis=1)
    compare_days.columns = ['Wednesday', 'Saturday']
    compare_days.plot(kind='bar', alpha=0.7, title='Orders by Hour: Wednesday vs Saturday')
    plt.xlabel('Hour of Day')
    plt.ylabel('Order Count')
    plt.show()

def plot_orders_per_customer(orders):
    print("\nAnalyzing orders per customer...")
    orders_per_customer = orders.groupby('user_id')['order_id'].count().sort_values()
    orders_per_customer.plot(kind='hist', bins=30, title='Distribution of Orders per Customer')
    plt.xlabel('Number of Orders')
    plt.show()

def plot_top_popular_products(order_products, products):
    print("\nIdentifying top 20 popular products...")
    # Merge only necessary columns to save memory
    top_products = order_products[['product_id']].merge(
        products[['product_id', 'product_name']], on='product_id', how='left'
    )
    top_products_counts = top_products['product_name'].value_counts().head(20)
    top_products_counts.plot(kind='bar', title='Top 20 Popular Products')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def plot_items_per_order(order_products):
    print("\nAnalyzing items per order...")
    items_per_order = order_products.groupby('order_id')['product_id'].count().value_counts().sort_index()
    items_per_order.head(30).plot(kind='bar', title='Items per Order Distribution (Top 30 sizes)')
    plt.xlabel('Number of Items')
    plt.show()

def plot_top_reordered_items(order_products, products):
    print("\nIdentifying top 20 reordered items...")
    reordered = order_products[order_products['reordered'] == 1]
    reordered_names = reordered[['product_id']].merge(
        products[['product_id', 'product_name']], on='product_id', how='left'
    )
    reordered_counts = reordered_names['product_name'].value_counts().head(20)
    reordered_counts.plot(kind='bar', title='Top 20 Reordered Items')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def plot_top_first_items(order_products, products):
    print("\nIdentifying top 20 items added to cart first...")
    first_items = order_products[order_products['add_to_cart_order'] == 1]
    first_items_names = first_items[['product_id']].merge(
        products[['product_id', 'product_name']], on='product_id', how='left'
    )
    first_items_counts = first_items_names['product_name'].value_counts().head(20)
    first_items_counts.plot(kind='bar', title="Top 20 'First in Cart' Items")
    plt.xticks(rotation=45, ha='right')
    plt.show()