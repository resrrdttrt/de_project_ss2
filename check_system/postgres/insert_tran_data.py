import pandas as pd
import psycopg2

def create_item_sales_table_if_not_exists(cursor):
    # Create the 'item_sales' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS item_sales (
        item_id VARCHAR(10),
        item_visibility FLOAT,
        store_id VARCHAR(10),
        item_store_sales FLOAT,
        PRIMARY KEY (item_id, store_id)
    );
    """
    cursor.execute(create_table_query)

def insert_item_sales_data_to_postgres(csv_file, db_config):
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
    create_item_sales_table_if_not_exists(cursor)

    # Insert each row into the PostgreSQL table
    for index, row in df.iterrows():
        cursor.execute("""
            INSERT INTO item_sales (item_id, item_visibility, store_id, item_store_sales)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (item_id, store_id) DO NOTHING;
        """, (row['Item_Id'], row['Item_Visibility'], row['Store_Id'], row['Item_Store_Sales']))

    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Item sales data inserted successfully.")


if __name__ == "__main__":
    # CSV file path
    csv_file = 'item_sales.csv'  # Replace with the path to your CSV file

    # PostgreSQL connection configuration
    db_config = {
        'host': 'localhost',       # Replace with your host
        'port': '5432',            # Default PostgreSQL port
        'database': 'postgres',     # Replace with your database name
        'user': 'postgres',       # Replace with your PostgreSQL username
        'password': 'postgres'  # Replace with your PostgreSQL password
    }

    # Insert item sales data into PostgreSQL
    insert_item_sales_data_to_postgres(csv_file, db_config)
