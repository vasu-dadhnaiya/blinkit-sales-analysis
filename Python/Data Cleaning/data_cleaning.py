import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATASETS
# =========================

orders = pd.read_csv("Main sales and delivery data/blinkit_orders.csv")
products = pd.read_csv("Product master information/blinkit_products.csv")
customers = pd.read_csv("Customer information/blinkit_customers.csv")
delivery = pd.read_csv("Delivery timing and status/blinkit_delivery_performance.csv")
feedback = pd.read_csv("Ratings and customer sentiment/blinkit_customer_feedback.csv")
inventory = pd.read_csv("Inventory stock tracking/blinkit_inventory.csv")
marketing = pd.read_csv("Marketing campaign performance/blinkit_marketing_performance.csv")
order_items = pd.read_csv("Product-wise order details/blinkit_order_items.csv")

# =========================
# DATA REVIEW
# =========================

print("\nOrders Dataset Preview")
print(orders.head())

print("\nOrders Info")
print(orders.info())

print("\nOrders Statistics")
print(orders.describe())

print("\nOrders Shape")
print(orders.shape)

# =========================
# CLEANING
# =========================

# Missing values
print("\nMissing Values")
print(orders.isnull().sum())

# Remove duplicates
orders.drop_duplicates(inplace=True)

# =========================
# DATE CONVERSION
# =========================

orders['order_date'] = pd.to_datetime(
    orders['order_date'],
    format='%d-%m-%Y %H:%M'
)

feedback['feedback_date'] = pd.to_datetime(
    feedback['feedback_date'],
    format='%d-%m-%Y'
)

orders['actual_delivery_time'] = pd.to_datetime(
    orders['actual_delivery_time'],
    format='%d-%m-%Y %H:%M'
)

orders['promised_delivery_time'] = pd.to_datetime(
    orders['promised_delivery_time'],
    format='%d-%m-%Y %H:%M'
)

# =========================
# FEATURE ENGINEERING
# =========================

# Delivery Delay in Minutes
orders['delivery_delay'] = (
    orders['actual_delivery_time']
    - orders['promised_delivery_time']
).dt.total_seconds() / 60

# Rating Category
def rating_category(rating):
    if rating >= 4:
        return "High"
    elif rating >= 2:
        return "Medium"
    else:
        return "Low"

feedback['rating_category'] = feedback['rating'].apply(rating_category)

# =========================
# ANALYSIS
# =========================

# Total Revenue
revenue = orders['order_total'].sum()

# Monthly Sales Trend
monthly_sales = orders.groupby(
    orders['order_date'].dt.month
)['order_total'].sum()

# =========================
# TOP CATEGORIES
# =========================

# Merge order items with products
category_sales = order_items.merge(
    products,
    on='product_id'
)

# Check available columns
print("\nOrder Items Columns:")
print(order_items.columns)

# Use correct sales column
# Replace 'subtotal' if your dataset uses another name

if 'subtotal' in category_sales.columns:
    top_categories = category_sales.groupby(
        'category'
    )['subtotal'].sum()

elif 'total_price' in category_sales.columns:
    top_categories = category_sales.groupby(
        'category'
    )['total_price'].sum()

elif 'price' in category_sales.columns:
    top_categories = category_sales.groupby(
        'category'
    )['price'].sum()

else:
    top_categories = category_sales.groupby(
        'category'
    ).size()

# =========================
# DELIVERY ANALYSIS
# =========================

Average_Delivery_Time = delivery[
    'delivery_time_minutes'
].mean()

delayed_deliveries = delivery[
    delivery['delivery_status'] == 'Delayed'
]

# =========================
# CUSTOMER ANALYSIS
# =========================

Top_Customers = orders.groupby(
    'customer_id'
)['order_total'].sum().sort_values(
    ascending=False
)

# =========================
# FEEDBACK ANALYSIS
# =========================

Sentiment_Distribution = feedback[
    'sentiment'
].value_counts()

# =========================
# MARKETING ANALYSIS
# =========================

marketing['conversion_rate'] = (
    marketing['conversions']
    / marketing['clicks']
) * 100

print("\nMarketing Columns:")
print(marketing.columns)

# =========================
# RESULTS
# =========================

print("\nMonthly Sales")
print(monthly_sales)

print("\nTotal Revenue")
print(revenue)

print("\nTop Categories")
print(top_categories.sort_values(ascending=False))

print("\nAverage Delivery Time")
print(Average_Delivery_Time)

print("\nDelayed Deliveries")
print(delayed_deliveries.shape[0])

print("\nTop Customers")
print(Top_Customers.head())

print("\nSentiment Distribution")
print(Sentiment_Distribution)

print("\nConversion Rates")
print(marketing['conversion_rate'].head())

# =========================
# VISUALIZATIONS
# =========================

# Monthly Sales Trend
plt.figure(figsize=(12, 6))

sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)

plt.show()

# =========================
# TOP CATEGORIES
# =========================

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_categories.values,
    y=top_categories.index
)

plt.title("Top Categories")
plt.xlabel("Revenue")
plt.ylabel("Category")

plt.show()

# =========================
# DELIVERY STATUS
# =========================

plt.figure(figsize=(6, 6))

delivery['delivery_status'].value_counts().plot.pie(
    autopct='%1.1f%%'
)

plt.title("Delivery Status")
plt.ylabel("")

plt.show()

# =========================
# CUSTOMER SENTIMENT
# =========================

plt.figure(figsize=(8, 5))

sns.countplot(
    x='sentiment',
    data=feedback
)

plt.title("Customer Sentiment")

plt.show()

# =========================
# MARKETING ROI
# =========================

# Check actual column names
print("\nMarketing Dataset Columns:")
print(marketing.columns)

# Replace column names if needed
# Example:
# x='spend'
# y='revenue'

if 'ad_spend' in marketing.columns and 'revenue_generated' in marketing.columns:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x='ad_spend',
        y='revenue_generated',
        data=marketing
    )

    plt.title("Marketing ROI")

    plt.show()

else:
    print("\nERROR:")
    print("Columns 'ad_spend' and/or 'revenue_generated' not found.")
    print("Please check actual column names above.")