import psycopg2
import random
from datetime import datetime, timedelta

# Connect to PostgreSQL (first database)
pg_conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="postgres",
    user="airflow",
    password="airflow"
)
pg_cursor = pg_conn.cursor()

# Connect to PostgreSQL (second database, replacing MySQL)
pg_conn_2 = psycopg2.connect(
    host="localhost",
    port="5432",
    database="postgres",
    user="postgres",
    password="postgres"
)
pg_cursor_2 = pg_conn_2.cursor()

schema_name = 'inventory'
pg_cursor_2.execute(f'SET search_path TO {schema_name};')
pg_conn_2.commit()

# Create the items table if it doesn't exist in the first PostgreSQL database (optional, with item_id as VARCHAR)
pg_cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        item_id VARCHAR(50) PRIMARY KEY,
        item_name VARCHAR(255) NOT NULL
    );
""")
pg_conn.commit()

# Create the amount table if it doesn't exist in the second PostgreSQL database
pg_cursor_2.execute("""
    CREATE TABLE IF NOT EXISTS amount (
        id SERIAL PRIMARY KEY,
        item_id VARCHAR(50) NOT NULL,
        amount INT NOT NULL,
        import_time TIMESTAMP NOT NULL
    );
""")

# Create the transaction table if it doesn't exist in the second PostgreSQL database
pg_cursor_2.execute("""
    CREATE TABLE IF NOT EXISTS transaction (
        id SERIAL PRIMARY KEY,
        item_id VARCHAR(50) NOT NULL,
        amount INT NOT NULL,
        buy_time TIMESTAMP NOT NULL,
        customer VARCHAR(255) NOT NULL
    );
""")
pg_conn_2.commit()

# Step 1: Select 100 random item_ids from the first PostgreSQL database
pg_cursor.execute("SELECT item_id FROM items ORDER BY RANDOM() LIMIT 100;")
item_ids = [row[0] for row in pg_cursor.fetchall()]

# Step 2: Insert 100 rows into the amount table in the second PostgreSQL database
for item_id in item_ids:
    amount = random.randint(0, 500)
    import_time = datetime.now()
    
    pg_cursor_2.execute("""
        INSERT INTO amount (item_id, amount, import_time)
        VALUES (%s, %s, %s);
    """, (item_id, amount, import_time))

# Step 3: Insert 1000 rows into the transaction table in the second PostgreSQL database
for _ in range(1000):
    item_id = random.choice(item_ids)
    amount = random.randint(1, 5)
    buy_time = datetime.now() - timedelta(days=random.randint(0, 30))  # Random date within the last 30 days
    customer = f"customer_{random.randint(1, 1000)}"  # Random customer name
    
    pg_cursor_2.execute("""
        INSERT INTO transaction (item_id, amount, buy_time, customer)
        VALUES (%s, %s, %s, %s);
    """, (item_id, amount, buy_time, customer))

# Commit the changes to the second PostgreSQL database
pg_conn_2.commit()

# Close all connections
pg_cursor.close()
pg_conn.close()
pg_cursor_2.close()
pg_conn_2.close()
