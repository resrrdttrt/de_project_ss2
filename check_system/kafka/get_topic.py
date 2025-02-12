from confluent_kafka.admin import AdminClient

def list_kafka_topics(broker_url):
    # Create a Kafka AdminClient
    admin_client = AdminClient({'bootstrap.servers': broker_url})

    # Get the list of topics
    try:
        metadata = admin_client.list_topics(timeout=10)
        topics = metadata.topics

        print("Available Kafka topics:")
        for topic in topics:
            print(f"- {topic}")

    except Exception as e:
        print(f"Failed to retrieve Kafka topics. Error: {e}")


if __name__ == "__main__":
    broker_url = 'localhost:9092'  # Replace with your Kafka broker URL

    # List all Kafka topics
    list_kafka_topics(broker_url)
