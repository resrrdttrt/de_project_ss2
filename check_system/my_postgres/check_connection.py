import psycopg2
from psycopg2 import OperationalError

def create_connection(db_name, user, password, host, port):
    """Create a database connection to a PostgreSQL database."""
    connection = None
    try:
        connection = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    finally:
        if connection:
            connection.close()

# # Replace these values with your actual database credentials
# db_name = 'postgres'
# user = 'postgres'
# password = 'postgres'
# host = 'localhost'
# port = '5432'

# Replace these values with your actual database credentials
db_name = 'postgres'
user = 'airflow'
password = 'airflow'
host = 'localhost'
port = '5433'

# Check connection
create_connection(db_name, user, password, host, port)