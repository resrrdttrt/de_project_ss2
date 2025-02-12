import pandas as pd
import psycopg2

def create_table_if_not_exists(cursor):
    # Create the 'stores' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS stores (
        store_id VARCHAR(10) PRIMARY KEY,
        store_establishment_year INT,
        store_size VARCHAR(50),
        store_location_type VARCHAR(50),
        store_type VARCHAR(50)
    );
    """
    cursor.execute(create_table_query)

def insert_data_to_postgres(csv_file, db_config):
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
    create_table_if_not_exists(cursor)

    # Insert each row into the PostgreSQL table
    for index, row in df.iterrows():
        cursor.execute("""
            INSERT INTO stores (store_id, store_establishment_year, store_size, store_location_type, store_type)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (store_id) DO NOTHING;
        """, (row['Store_Id'], row['Store_Establishment_Year'], row['Store_Size'], row['Store_Location_Type'], row['Store_Type']))

    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Data inserted successfully.")


if __name__ == "__main__":
    # CSV file path
    csv_file = 'dataset/Stores.csv'  # Replace with the path to your CSV file

    # PostgreSQL connection configuration
    db_config = {
        'host': 'localhost',       # Replace with your host
        'port': '5432',            # Default PostgreSQL port
        'database': 'postgres',     # Replace with your database name
        'user': 'postgres',       # Replace with your PostgreSQL username
        'password': 'postgres'  # Replace with your PostgreSQL password
    }

    # Insert data into PostgreSQL
    insert_data_to_postgres(csv_file, db_config)
