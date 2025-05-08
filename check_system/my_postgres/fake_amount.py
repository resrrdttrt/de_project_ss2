import uuid
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker for generating fake data
fake = Faker()

# Number of stores and items
NUM_STORES = 5
NUM_ITEMS = 20
ITEMS_PER_STORE = 10

# Helper function to generate random store_id and item_id
def generate_store_id(index):
    return f"STORE{index:03d}"

def generate_item_id(index):
    return f"ITEM{index:03d}"

# Generate fake data for stores table
stores = []
for i in range(1, NUM_STORES + 1):
    store = {
        'store_id': generate_store_id(i),
        'store_establishment_year': random.randint(1980, 2023),
        'store_size': random.choice(['Small', 'Medium', 'Large']),
        'store_location_type': random.choice(['Urban', 'Suburban', 'Rural']),
        'store_type': random.choice(['Retail', 'Warehouse', 'Outlet'])
    }
    stores.append(store)

# Generate fake data for items table
items = []
for i in range(1, NUM_ITEMS + 1):
    item = {
        'item_id': generate_item_id(i),
        'item_fit_type': random.choice(['Regular', 'Slim', 'Loose', 'Custom']),
        'item_fabric': random.choice(['Cotton', 'Polyester', 'Wool', 'Silk', 'Linen']),
        'item_fabric_amount': round(random.uniform(0.5, 5.0), 2),  # Fabric amount in meters
        'item_mrp': round(random.uniform(10.0, 200.0), 2)  # Price in dollars
    }
    items.append(item)

# Generate fake data for amount table
amounts = []
for store in stores:
    # Randomly select 10 items for each store
    selected_items = random.sample(items, ITEMS_PER_STORE)
    for item in selected_items:
        amount = {
            'id': str(uuid.uuid4()),  # Generate UUID
            'item_id': item['item_id'],
            'amount': random.randint(10000000, 100000000),  # Stock quantity
            'import_time': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S'),
            'store_id': store['store_id']
        }
        amounts.append(amount)

# # Print SQL INSERT statements (or use psycopg2 to insert into PostgreSQL)
# print("INSERT INTO public.stores (store_id, store_establishment_year, store_size, store_location_type, store_type) VALUES")
# for i, store in enumerate(stores):
#     print(f"('{store['store_id']}', {store['store_establishment_year']}, '{store['store_size']}', "
#           f"'{store['store_location_type']}', '{store['store_type']}')", end="")
#     print(";" if i == len(stores) - 1 else ",")

# print("\nINSERT INTO public.items (item_id, item_fit_type, item_fabric, item_fabric_amount, item_mrp) VALUES")
# for i, item in enumerate(items):
#     print(f"('{item['item_id']}', '{item['item_fit_type']}', '{item['item_fabric']}', "
#           f"{item['item_fabric_amount']}, {item['item_mrp']})", end="")
#     print(";" if i == len(items) - 1 else ",")

# print("\nINSERT INTO public.amount (id, item_id, amount, import_time, store_id) VALUES")
# for i, amount in enumerate(amounts):
#     print(f"('{amount['id']}', '{amount['item_id']}', {amount['amount']}, "
#           f"'{amount['import_time']}', '{amount['store_id']}')", end="")
#     print(";" if i == len(amounts) - 1 else ",")

# Optional: Connect to PostgreSQL and insert data

import psycopg2

# Database connection parameters
db_params = {
    "dbname": "postgres",
    "user": "airflow",
    "password": "airflow",
    "host": "localhost",
    "port": "5433"
}

# Connect to PostgreSQL
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# # Insert stores
# for store in stores:
#     cursor.execute(
#         "INSERT INTO public.stores (store_id, store_establishment_year, store_size, store_location_type, store_type) "
#         "VALUES (%s, %s, %s, %s, %s)",
#         (store['store_id'], store['store_establishment_year'], store['store_size'],
#          store['store_location_type'], store['store_type'])
#     )

# # Insert items
# for item in items:
#     cursor.execute(
#         "INSERT INTO public.items (item_id, item_fit_type, item_fabric, item_fabric_amount, item_mrp) "
#         "VALUES (%s, %s, %s, %s, %s)",
#         (item['item_id'], item['item_fit_type'], item['item_fabric'],
#          item['item_fabric_amount'], item['item_mrp'])
#     )

# Insert amounts
for amount in amounts:
    cursor.execute(
        "INSERT INTO public.amount (id, item_id, amount, import_time, store_id) "
        "VALUES (%s, %s, %s, %s, %s)",
        (amount['id'], amount['item_id'], amount['amount'],
         amount['import_time'], amount['store_id'])
    )

# Commit and close
conn.commit()
cursor.close()
conn.close()
