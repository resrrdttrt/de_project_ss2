import psycopg2
import json
import datetime

# Database connection parameters for the target PostgreSQL database
target_db_config = {
    'dbname': 'postgres',
    'user': 'airflow',
    'password': 'airflow',
    'host': 'localhost',
    'port': '5433'
}

# Read the data from the file
input_file = 'tran_output.json'
with open(input_file, 'r') as f:
    data = json.load(f)

columns = data['columns']
rows = data['data']

# Function to convert ISO 8601 string back to datetime
def convert_to_datetime(obj):
    if isinstance(obj, str):
        try:
            # Attempt to parse ISO 8601 date string
            return datetime.datetime.fromisoformat(obj)
        except ValueError:
            return obj  # If it can't be parsed as datetime, return the string unchanged
    return obj

# Convert ISO 8601 strings back to datetime objects where applicable
rows = [[convert_to_datetime(value) for value in row] for row in rows]

# Connect to the target database
conn = psycopg2.connect(**target_db_config)
cursor = conn.cursor()

# Create table query (using the column names from the source table)
target_table = "transaction2"  # Replace with your target table name

# Create the target table if it doesn't exist
create_table_query = f"""
CREATE TABLE IF NOT EXISTS {target_table} (
    {', '.join([f"{col} TEXT" for col in columns])}
);
"""

cursor.execute(create_table_query)

# Prepare the insert query
insert_query = f"""
INSERT INTO {target_table} ({', '.join(columns)})
VALUES ({', '.join(['%s' for _ in columns])});
"""

# Insert all rows from the file into the target table
for row in rows:
    cursor.execute(insert_query, row)

# Commit the transaction
conn.commit()

print(f"Data has been inserted into {target_table}")

# Close the connections
cursor.close()
conn.close()
