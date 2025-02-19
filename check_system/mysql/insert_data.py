import psycopg2
import mysql.connector
import random
from datetime import datetime, timedelta

# Connect to PostgreSQL
pg_conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="postgres",
    user="airflow",
    password="airflow"
)
pg_cursor = pg_conn.cursor()

# Create the items table if it doesn't exist in PostgreSQL (optional, with item_id as VARCHAR)
pg_cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        item_id VARCHAR(50) PRIMARY KEY,
        item_name VARCHAR(255) NOT NULL
    );
""")
pg_conn.commit()

# Connect to MySQL
mysql_conn = mysql.connector.connect(
    host="localhost",
    database="inventory",
    user="mysqluser",
    password="mysqlpw"
)
mysql_cursor = mysql_conn.cursor()

# Create the amount table if it doesn't exist in MySQL (with item_id as VARCHAR)
mysql_cursor.execute("""
    CREATE TABLE IF NOT EXISTS amount (
        id INT AUTO_INCREMENT PRIMARY KEY,
        item_id VARCHAR(50) NOT NULL,
        amount INT NOT NULL,
        import_time DATETIME NOT NULL
    );
""")

# Create the transaction table if it doesn't exist in MySQL (with item_id as VARCHAR)
mysql_cursor.execute("""
    CREATE TABLE IF NOT EXISTS transaction (
        id INT AUTO_INCREMENT PRIMARY KEY,
        item_id VARCHAR(50) NOT NULL,
        amount INT NOT NULL,
        buy_time DATETIME NOT NULL,
        customer VARCHAR(255) NOT NULL
    );
""")
mysql_conn.commit()

# Step 1: Select 100 random item_ids from PostgreSQL
pg_cursor.execute("SELECT item_id FROM items ORDER BY RANDOM() LIMIT 100;")
item_ids = [row[0] for row in pg_cursor.fetchall()]

# Step 2: Insert 100 rows into amount table in MySQL
for item_id in item_ids:
    amount = random.randint(0, 500)
    import_time = datetime.now()
    
    mysql_cursor.execute("""
        INSERT INTO amount (item_id, amount, import_time)
        VALUES (%s, %s, %s);
    """, (item_id, amount, import_time))

# Step 3: Insert 1000 rows into transaction table in MySQL
for _ in range(1000):
    item_id = random.choice(item_ids)
    amount = random.randint(1, 5)
    buy_time = datetime.now() - timedelta(days=random.randint(0, 30))  # Random date within the last 30 days
    customer = f"customer_{random.randint(1, 1000)}"  # Random customer name
    
    mysql_cursor.execute("""
        INSERT INTO transaction (item_id, amount, buy_time, customer)
        VALUES (%s, %s, %s, %s);
    """, (item_id, amount, buy_time, customer))

# Commit the changes to MySQL
mysql_conn.commit()

# Close all connections
pg_cursor.close()
pg_conn.close()
mysql_cursor.close()
mysql_conn.close()
