import pandas as pd
import psycopg2

def create_items_table_if_not_exists(cursor):
    # Create the 'items' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS items (
        item_id VARCHAR(10) PRIMARY KEY,
        item_fit_type VARCHAR(50),
        item_fabric VARCHAR(50),
        item_fabric_amount FLOAT,
        item_mrp FLOAT
    );
    """
    cursor.execute(create_table_query)

def insert_items_data_to_postgres(csv_file, db_config):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Connect to PostgreSQL database
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )
    cursor = conn.cursor()

    # Create table if it doesn't exist
    create_items_table_if_not_exists(cursor)

    # Insert each row into the PostgreSQL table
    for index, row in df.iterrows():
        cursor.execute("""
            INSERT INTO items (item_id, item_fit_type, item_fabric, item_fabric_amount, item_mrp)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (item_id) DO NOTHING;
        """, (row['Item_Id'], row['Item_Fit_Type'], row['Item_Fabric'], row['Item_Fabric_Amount'], row['Item_MRP']))

    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Item data inserted successfully.")


if __name__ == "__main__":
    # CSV file path
    csv_file = 'items.csv'  # Replace with the path to your CSV file

    # PostgreSQL connection configuration
    db_config = {
        'host': 'localhost',       # Replace with your host
        'port': '5432',            # Default PostgreSQL port
        'database': 'postgres',     # Replace with your database name
        'user': 'postgres',       # Replace with your PostgreSQL username
        'password': 'postgres'  # Replace with your PostgreSQL password
    }

    # Insert item data into PostgreSQL
    insert_items_data_to_postgres(csv_file, db_config)
