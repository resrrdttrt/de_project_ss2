from kafka import KafkaProducer
from kafka.errors import KafkaError

def check_kafka_connection(broker_url):
    try:
        # Create a Kafka producer instance
        producer = KafkaProducer(bootstrap_servers=[broker_url])
        
        # Attempt to send a test message
        future = producer.send('test-topic', b'Test connection')
        future.get(timeout=10)  # Block for 'send' result for up to 10 seconds

        print(f"Connected to Kafka broker at {broker_url} successfully.")
        producer.close()

    except KafkaError as e:
        print(f"Failed to connect to Kafka broker at {broker_url}. Error: {e}")

if __name__ == "__main__":
    broker_url = 'localhost:9092'  # Replace with your Kafka broker URL
    check_kafka_connection(broker_url)
