import psycopg2
import json
import datetime

# Database connection parameters for the source PostgreSQL database
source_db_config = {
    'dbname': 'postgres',
    'user': 'airflow',
    'password': 'airflow',
    'host': 'localhost',
    'port': '5433'
}

# Connect to the source database
conn = psycopg2.connect(**source_db_config)
cursor = conn.cursor()

# Query to fetch all data from the source table
source_table = "transaction"
select_query = f"SELECT * FROM {source_table};"

cursor.execute(select_query)

# Fetch all rows from the result
data = cursor.fetchall()

# Get column names (this is useful when creating the target table later)
column_names = [desc[0] for desc in cursor.description]

# Function to convert datetime objects to strings
def convert_datetime(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()  # Convert to ISO 8601 string
    return obj  # Return the object unchanged if it's not a datetime

# Convert datetime objects in the data to strings
data = [[convert_datetime(value) for value in row] for row in data]

# Save data to a file (JSON format for easy readability and use)
output_file = 'tran_output.json'
with open(output_file, 'w') as f:
    json.dump({'columns': column_names, 'data': data}, f, indent=2)

print(f"Data has been saved to {output_file}")

# Close the connections
cursor.close()
conn.close()
