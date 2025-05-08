import uuid
import random
import datetime
import psycopg2
from faker import Faker

# Initialize Faker
fake = Faker()

# Database connection parameters
DB_PARAMS = {
    "dbname": "postgres",
    "user": "airflow",
    "password": "airflow",
    "host": "localhost",
    "port": "5433"
}

# Store IDs - you can customize these
STORE_IDS = ["STORE002", "STORE001", "STORE003", "STORE004", "STORE005"]

# We'll fetch real item IDs from the database

def get_item_ids_from_db():
    """Fetch actual item IDs from the items table"""
    item_ids = []
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        cursor.execute("SELECT item_id FROM public.items")
        item_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error fetching item IDs: {e}")
        # Fallback to some sample item IDs if DB connection fails
        item_ids = [f"item{str(i).zfill(3)}" for i in range(1, 10)]
    
    if not item_ids:
        # If no items found, create some sample IDs
        item_ids = [f"item{str(i).zfill(3)}" for i in range(1, 10)]
    
    return item_ids

def generate_transaction_data(num_records=10000):
    """Generate fake transaction data with increasing daily amounts"""
    # Get actual item IDs from the database
    item_ids = get_item_ids_from_db()
    print(f"Retrieved {len(item_ids)} item IDs from database")
    
    # Calculate date range (1 month ago to 1 month in the future)
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=30)  # 1 month ago
    end_date = now + datetime.timedelta(days=30)    # 1 month in the future
    total_days = (end_date - start_date).days
    
    # Calculate how many transactions per day (roughly)
    transactions_per_day = num_records // total_days
    
    all_transactions = []
    
    # Base amount that will increase day by day
    base_amount_per_day = 3000
    amount_increase_per_day = 100
    
    # Generate data for each day
    for day_offset in range(total_days):
        current_date = start_date + datetime.timedelta(days=day_offset)
        
        # Increase base amount each day
        daily_base_amount = base_amount_per_day + (day_offset * amount_increase_per_day)
        
        # Generate transactions for this day
        daily_transactions = []
        for _ in range(transactions_per_day + random.randint(-5, 5)):  # Add some randomness
            # Generate random time within the current day
            hour = random.randint(8, 22)  # Business hours
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = current_date.replace(
                hour=hour, 
                minute=minute, 
                second=second
            )
            
            # Generate transaction
            transaction = {
                "id": str(uuid.uuid4()),
                "item_id": random.choice(item_ids),
                "amount": random.randint(10, 500) + (day_offset * 2),  # Individual transaction amounts also increase slightly
                "buy_time": timestamp,
                "customer": fake.name(),
                "store_id": random.choice(STORE_IDS)
            }
            
            daily_transactions.append(transaction)
        
        # Add daily transactions to all transactions
        all_transactions.extend(daily_transactions)
    
    # Ensure we have approximately the requested number of records
    remaining = num_records - len(all_transactions)
    if remaining > 0:
        print(f"Generating {remaining} additional transactions to reach target...")
        
        for _ in range(remaining):
            # Generate a random day within our date range
            random_days = random.randint(0, total_days - 1)
            current_date = start_date + datetime.timedelta(days=random_days)
            
            hour = random.randint(8, 22)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = current_date.replace(hour=hour, minute=minute, second=second)
            
            transaction = {
                "id": str(uuid.uuid4()),
                "item_id": random.choice(item_ids),
                "amount": random.randint(10, 500) + (random_days * 2),
                "buy_time": timestamp,
                "customer": fake.name(),
                "store_id": random.choice(STORE_IDS)
            }
            
            all_transactions.append(transaction)
    
    return all_transactions

def insert_transactions_to_db(transactions):
    """Insert transactions into the database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        for transaction in transactions:
            cursor.execute("""
                INSERT INTO public.transaction (id, item_id, amount, buy_time, customer, store_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                transaction["id"],
                transaction["item_id"],
                transaction["amount"],
                transaction["buy_time"],
                transaction["customer"],
                transaction["store_id"]
            ))
        
        conn.commit()
        print(f"Successfully inserted {len(transactions)} transactions.")
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def print_transaction_stats(transactions):
    """Print statistics about the generated transactions"""
    # Group by day
    days = {}
    for transaction in transactions:
        day = transaction["buy_time"].date()
        if day not in days:
            days[day] = []
        days[day].append(transaction)
    
    print(f"Generated {len(transactions)} transactions across {len(days)} days")
    
    # Print weekly statistics to avoid too much output
    weeks = {}
    for day, day_transactions in days.items():
        # Get the week number
        week = day.isocalendar()[1]  # ISO week number
        year = day.year
        week_key = f"{year}-W{week}"
        
        if week_key not in weeks:
            weeks[week_key] = []
        weeks[week_key].extend(day_transactions)
    
    # Print weekly statistics
    print("\nWeekly Statistics:")
    for week_key, week_transactions in sorted(weeks.items()):
        total_amount = sum(t["amount"] for t in week_transactions)
        avg_daily = total_amount / 7  # Approximate daily average
        print(f"Week {week_key}: {len(week_transactions)} transactions, total amount: ${total_amount}, avg daily: ${avg_daily:.2f}")
        
    # Print some sample days to show the daily progression
    print("\nSample Daily Statistics (showing 3 days from each week):")
    sorted_days = sorted(days.items())
    sample_indices = []
    
    # Take 3 days from beginning, middle, and end
    for i in range(3):
        if i < len(sorted_days):
            sample_indices.append(i)
    
    middle_start = len(sorted_days) // 2 - 1
    for i in range(3):
        if middle_start + i < len(sorted_days):
            sample_indices.append(middle_start + i)
    
    end_start = len(sorted_days) - 3
    for i in range(3):
        if end_start + i < len(sorted_days):
            sample_indices.append(end_start + i)
    
    for i in sample_indices:
        day, day_transactions = sorted_days[i]
        total_amount = sum(t["amount"] for t in day_transactions)
        print(f"Day {day}: {len(day_transactions)} transactions, total amount: ${total_amount}")


def save_to_csv(transactions, filename="transactions.csv"):
    """Save transactions to a CSV file"""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["id", "item_id", "amount", "buy_time", "customer", "store_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for transaction in transactions:
            writer.writerow(transaction)
    
    print(f"Saved transactions to {filename}")

if __name__ == "__main__":
    # Generate transaction data (10,000 records by default)
    transactions = generate_transaction_data(10000)
    
    # Print statistics
    print_transaction_stats(transactions)
    
    # Save to CSV (optional)
    # save_to_csv(transactions)
    
    # Insert into database (uncomment to execute)
    insert_transactions_to_db(transactions)
    
    # Example: To test without inserting into DB, you can print a few sample transactions
    print("\nSample transactions:")
    for i in range(5):
        transaction = transactions[i]
        print(f"ID: {transaction['id']}, Item: {transaction['item_id']}, Amount: ${transaction['amount']}, "
              f"Time: {transaction['buy_time']}, Customer: {transaction['customer']}, Store: {transaction['store_id']}")
              
    print("\nTo insert data into the database, uncomment the insert_transactions_to_db() call above")