import pandas as pd
import matplotlib.pyplot as plt

# datasets
orders         = pd.read_csv('/datasets/instacart_orders.csv', sep=';')
products       = pd.read_csv('/datasets/products.csv', sep=';')
departments    = pd.read_csv('/datasets/departments.csv', sep=';')
aisles         = pd.read_csv('/datasets/aisles.csv', sep=';')
order_products = pd.read_csv('/datasets/order_products.csv', sep=';')

# explore
print(orders.head())
print(products.head())
print(departments.head())
print(aisles.head())
print(order_products.head())

dfs = {
    'orders': orders,
    'products': products,
    'departments': departments,
    'aisles': aisles,
    'order_products': order_products
}

print("Review data:\n")
for name, df in dfs.items():
    print(f"{name}.info()\n")
    print(df.info())
    print(f"{name}.head()\n")
    print(df.head())
    print(f"{name}.describe()\n")
    print(df.describe(include='all'))
    print("\n" + "-"*50 + "\n")



#Review each col for mislabeled missing data
print("Review missing data:\n")
for name, df in dfs.items():
    total_rows = df.shape[0]
    for col in df.columns:
        non_missing_count = df[col].count()
        missing_count = total_rows - non_missing_count
        print(f"Location: {name}[{col}]")
        print(f"  Total rows: {total_rows}")
        print(f"  Non-missing values: {non_missing_count}\n")
        print(f"  Missing values: {missing_count}")
        print(f"  .isnull() count: {df[col].isnull().sum()}\n")
    print("\n" + "-"*50 + "\n")

'''
only orders, products and order_products have missing values.
'''

#MCAR analysis
print("MCAR analysis")
cols_with_missing_values = {
    'orders': ['days_since_prior_order'],
    'products': ['product_name'],
    'order_products': ['add_to_cart_order']
}

for name, df in dfs.items():
    if name in cols_with_missing_values:
        for col in cols_with_missing_values[name]:
            if col in df.columns: #safety check
                print(f"{name}[{col}] missing rows:")
                print(df[df[col].isna()])
                print("\n")
            else:
                print("\n")

# Display rows where the product_name column has missing values
products[products['product_name'].isna()]

'''
all rows with missing product_name might be aisle_id 100 and department_id 21.
'''

# Combine conditions to check for missing product names in aisles other than 100
missing_products_product_name_not_aisle_100 = (
    products
    .loc[
        (products['product_name'].isna()) &
        (products['aisle_id'] != 100)])

print(missing_products_product_name_not_aisle_100)

# Combine conditions to check for missing product names in aisles other than 21
missing_products_product_name_not_dept_21 = (
    products
    .loc[
        (products['product_name'].isna()) & 
        (products['department_id'] != 21)])

print(missing_products_product_name_not_dept_21)

# Checking what this aisle and department is.
print(aisles
      .loc[
          (aisles['aisle_id'] > 90) & (aisles['aisle_id'] < 110),
          ['aisle_id','aisle']])

print(departments.loc[
    (departments['department_id'] >= 15) & (departments['department_id'] <= 25) ,
    ['department_id','department']])

# Fill missing product names with 'Unknown'
products['product_name'].fillna("Unknown",inplace = True)
print(products.loc[
    products['aisle_id'] == 100,
    ['aisle_id','product_name']])

# orders:
# Display rows where the days_since_prior_order column has missing values
print(orders
      [orders['days_since_prior_order'].isna()]
      .sample(30,random_state=1))

# Are there any missing values where it's not a customer's first order?
print(orders[
    (orders['days_since_prior_order'].isna()) & (orders['order_number'] != 1)])
'''
Nan on days_since_prior_order when it's their first order makes sense.
'''

# missing values from the order_products table:
# Rows where the add_to_cart_order column has missing values
print(order_products[order_products['add_to_cart_order'].isna()])
    
# Save all order IDs with at least one missing value in 'add_to_cart_order'
#numpy array of order_ids
order_ids_with_missing_add_to_cart_order = (
    order_products[order_products['add_to_cart_order'].isna()]['order_id'].unique())

#df of order_ids where add_to_cart_order is missing
order_products_with_missing_add_to_cart_order = (
    order_products[order_products['order_id'].isin(order_ids_with_missing_add_to_cart_order)])

# Size of orders with missing values:
order_sizes = (
    order_products_with_missing_add_to_cart_order.groupby('order_id').size().sort_values())
print("order_sizes: \n",order_sizes.describe()) 
'''
min order is 65
'''

# Replace missing values with 999 and convert column to integer type
#test replace
print(order_products['add_to_cart_order'].isna().sum())
test = order_products.copy()
test['add_to_cart_order'] = order_products['add_to_cart_order'].fillna(999, inplace = False)
print(test['add_to_cart_order'].isna().sum())

#apply replace
order_products['add_to_cart_order'] = order_products['add_to_cart_order'].fillna(999, inplace = False)
print(order_products['add_to_cart_order'].isna().sum())

#convert column to type int
order_products['add_to_cart_order'] = order_products['add_to_cart_order'].astype(int)
print(order_products['add_to_cart_order'].dtype)

'''
For some reason, any item placed in the cart 65th or later has a missing value in the 'add_to_cart_order' column. Maybe the data type of that column in the database could only hold integer values from 1 to 64. We've decided to replace the missing values with a code value, 999, that represents an unknown placed in cart order above 64. We also converted the column to integer data type. We just need to be careful to remember this if we perform calculations using this column during our analysis.

Other sensible code values we could've used are 0 or -1 because they don't show up elsewhere in the dataset and they don't have any real physical meaning for this variable.

Also note that, for orders with exactly 65 items, we could replace the missing value with 65. But we're going to neglect that for now since we can't determine the 65th item for all orders with 66 items or more.
'''

#orders:
# find duplicates and remove
print(orders.duplicated().sum())

# View the duplicate rows
print(orders[orders.duplicated()])

# Remove duplicate orders
orders = orders.drop_duplicates().reset_index(drop=True)

# Double check for duplicate rows
print(orders.duplicated().sum())

# products:
# Check for fully duplicate rows
products.duplicated().sum()

# Check for just duplicate product IDs
products['product_id'].duplicated().sum()

# Check for just duplicate product names
products['product_name'].str.lower().duplicated().sum()

# duplicate found
products[products['product_name'].str.lower() == 'high performance energy drink']

# Drop duplicate product names (case insensitive)
# start value
print(products['product_name'].shape[0])

# make all values lower case
products['product_name'] = (
    products['product_name']
    .str
    .lower())

# drop duplicates
products.drop_duplicates(subset='product_name', inplace=True)

# end value
print(products['product_name'].shape[0])

# departments df
# duplicate entries in the departments dataframe
departments.duplicated().sum() # 0

# Check for aisles entries in the departments dataframe
print(departments) #none

# Check for duplicate entries in the order_products dataframe
order_products.duplicated().sum() # 0

# verify that col values make sense

# Verify that the 'order_hour_of_day' are sensible
print(sorted(orders['order_hour_of_day'].unique())) # 0-23
print(orders['order_hour_of_day'].nunique()) # 24

#Verify that`'order_dow'` values are sensible
print(sorted(orders['order_dow'].unique())) # 0-6
print(orders['order_dow'].nunique()) # 7

# Time of day people shop for groceries:
# orders by hour
order_by_hour = (
    orders['order_hour_of_day']
    .value_counts()
    .sort_index())
print(order_by_hour)

#plot
order_by_hour.plot(
    x=order_by_hour.index,
    y='order_id', 
    kind='bar'
)

plt.title('Time of Day People shop for groceries')
plt.xlabel('Hour in the day')
plt.ylabel('Number of orders')
plt.show()

# Day of the week people shop for groceries:
#orders by day
orders_per_day = (
    orders['order_dow']
    .value_counts()
    .sort_index())
print(orders_per_day)

#plot
orders_per_day.plot(x=orders_per_day.index,y='order_dow', kind='bar')
'''
The data dictionary does not state which integer corresponds to which day of the week. 
Assuming Sunday = 0, then people place more orders at the beginning of the week (Sunday and Monday).
'''

# Days between orders:
#view
print("days_since_prior_order.describe: \n",
      orders['days_since_prior_order'].describe(),
      "\n")

#number of orders by days_since_prior_order
print("days_since_prior_order.value counts: \n",
      orders['days_since_prior_order']
      .value_counts()
      .sort_index(),
      "\n")

orders_per_days_since_prior_order = (
    orders['days_since_prior_order']
    .value_counts()
    .sort_index())

#plot
orders_per_days_since_prior_order.plot(
    x=orders_per_days_since_prior_order.index,
    y='days_since_prior_order',
    kind='bar'
)
'''
The 0 values probably correspond to customers who placed more than one order on the same day.

The max value of 30 days and the high spike at that value is puzzling though. The spike might be explained by people who set up recurring subscriptions to automatically order once a month. But that doesn't explain why there are no values above 30 days. I would expect many customers to place orders less often than once a month. Maybe those customers were intentionally excluded from the dataset.
Disregarding the spike at 30 days, most people wait between 2 to 10 days in between orders. The most common wait time is 7 days. In other words, it's common for people to place weekly grocery orders. Interestingly, in the tail of the distribution we also see small spikes at 14, 21, and 28 days. These would correspond to orders every 2, 3, or 4 weeks.
'''

#Is there a difference in 'order_hour_of_day' distributions on Wednesdays and Saturdays:
#Wednesday orders
Wednesday_orders = orders[orders['order_dow'] == 3]
Wednesday_orders_per_hour_of_day = (
    Wednesday_orders['order_hour_of_day']
    .value_counts()
    .sort_index())

#Saturday orders
Saturday_orders = orders[orders['order_dow'] == 6]
Saturday_orders_per_hour_of_day = (
    Saturday_orders['order_hour_of_day']
    .value_counts()
    .sort_index())

# make Wednesday Saturday df
order_hour_of_day_Wednesdays_Saturdays = (
    pd.concat([Wednesday_orders_per_hour_of_day,
               Saturday_orders_per_hour_of_day],
              axis = 1))
order_hour_of_day_Wednesdays_Saturdays.columns = ['Wednesday_orders','Saturday_orders']
print(order_hour_of_day_Wednesdays_Saturdays)

# plot
order_hour_of_day_Wednesdays_Saturdays['Wednesday_orders'].plot(
    kind='hist',
    #bins=30,
    label='Wednesday_orders',  
    alpha=0.5  
)

order_hour_of_day_Wednesdays_Saturdays['Saturday_orders'].plot(
    kind='hist',
    #bins=30,
    label='Saturday_orders',  
    alpha=0.5  
)

plt.legend()
plt.show()

# Distribution of number of orders per customer
# group order by user_id and sort
customers_number_of_orders = (
    orders
    .groupby('user_id')['order_id']
    .count()  
    .sort_values(ascending = True))
print((customers_number_of_orders))

# plot
customers_number_of_orders.plot(
    kind = 'hist',
    bins = 30 #since there are less than 30 numbers, I wanted to see more detail.
)

# top 20 popular products:
# Merge order_products and products
order_products_names = (
    order_products[['order_id', 'product_id']]
    .merge(products[['product_id', 'product_name']],
           on='product_id',
           how='left')
)

# Group by both product_id and product_name and count
ordered_products_count = (
    order_products_names
    .groupby(['product_id', 'product_name'])
    .size()
    .sort_values(ascending=False)
)

# View the top 20 products
print(ordered_products_count.head(20))

# plot top 20
ordered_products_count.iloc[:20].plot(
    kind = 'bar'
)
plt.xticks(rotation=45, ha='right') 
plt.show()

# number of items bought per order
products_ids_value_counts = (
    order_products[['order_id','product_id']]
    .groupby('order_id')['product_id'] 
    .count()   
    .value_counts()  
    .sort_index() 
)

#plot
products_ids_value_counts.plot(
    kind='bar',
    title='Distribution of Number of Products per Order',
    xlabel='Number of Products per Order',
    ylabel='Number of Orders'
)
plt.show()

'''Most of the order numbers are in the tail of the distribution. '''

#plot
products_ids_value_counts[:30].plot(
    kind='bar',
    #bins=10,
    #edgecolor='black',
    title='Distribution of Number of Products per Order',
    xlabel='Number of Products per Order',
    ylabel='Number of Orders',
    #grid=True
)
plt.show()

# top 20 reordered items
# filter
reordered_products = order_products[order_products['reordered'] == 1]
print(reordered_products.describe())

#merge reordered_products and products
reordered_products = (reordered_products
                      .merge(
                          products[['product_name','product_id']],
                          on = 'product_id',
                          how = 'left',
                      ))
print(reordered_products)

#groupby: product_id' , 'product_name
reordered_products_count = (
    reordered_products
    .groupby(['product_id' , 'product_name']) 
    .size() 
    .sort_values(ascending=False))  
print(reordered_products_count.head())

# visualize the top 20 reordered products
reordered_products_count[:20].plot(
    kind='bar',
    x='product_name',
    y='count',
    legend=False,
    figsize=(12, 6)
)
plt.xticks(rotation=45, ha='right')
plt.show()

# For each product, proportion that are reorders:
#Merge the datasets: Combine order_products with the products dataset to access product names and IDs in the same DataFrame.
order_products_with_product_names = (
    order_products.merge(
        products[['product_id','product_name']], #only merges common key, and the col I want.
        on = 'product_id', #key
        how = 'left' #keeps left, just adds new col. 
    ))
print(order_products_with_product_names.head())

#Group the data: Group by product_id and product_name to isolate each product's order history.
agg_dict = {'reordered': 'mean'} 

product_names_product_id = (
    order_products_with_product_names
    .groupby(['product_name'])
    .agg(agg_dict) 
    .sort_values(by = 'reordered' , ascending = False) 
    .reset_index())
print(product_names_product_id['reordered'].describe())

#Link order and customer information.
agg_dict = {'reordered':'mean'}
orders_with_order_products = (
    orders[['order_id','user_id']]
    .merge(
        order_products[['order_id','product_id','reordered']],
        on = 'order_id', 
        how = 'left'
    )
    .groupby('user_id')
    .agg(agg_dict)['reordered'] 
    .sort_values(ascending = False)
    .reset_index()
)
print(orders_with_order_products)

# plot
orders_with_order_products['reordered'].plot(
    kind='hist',
    x='reordered')

plt.title('Histogram of Reorder ratios')
plt.xlabel('Ratio of Reorders')
plt.ylabel('Ratio of Reorders count')
plt.show()

# top 20 'first' items in the cart:
# Link product names and IDs.
order_products_and_products = (
    order_products[['product_id', 'order_id','add_to_cart_order']] #what we need
    .merge(
        products[['product_id' , 'product_name']], #what we need
        on = 'product_id', #key
        how = 'left' #left merge
    )
)
print(order_products_and_products)

#Filter: add_to_cart_order equals 1, indicating the first item added to the cart.
agg_dict = {'product_name': 'count'}
order_products_and_products = (
    order_products_and_products[order_products_and_products['add_to_cart_order'] == 1][['product_name','add_to_cart_order']]
    .groupby('product_name') #group 
    .agg(agg_dict)['product_name'] #count product_name
    .sort_values(ascending = False) #sort
    .reset_index(name = 'count') #turn series to df, name index 'count'
)
print(order_products_and_products.head())

#plot
order_products_and_products[:20].plot(
    kind='bar',
    x='product_name',
    y='count')

plt.xticks(rotation=45, ha='right') 
plt.show()

'''
The products that are most often placed into the cart first are produce, dairy, and beverages such as soda or water. 
I couldn't really say why that is without experience using Instacart because this could have more to do with app design than properties of the products. 
I do notice that there is considerable overlap between this result and the previous result for most popular and most reordered item types. 
It could simply be that the app prioritizes popular items as the first suggested purchases, so it happens to be more convenient for customers to place these items in their cart first.
'''