from confluent_kafka import Consumer, KafkaException

def consume_messages(broker_url, topic_name, group_id):
    # Create a Kafka Consumer
    consumer = Consumer({
        'bootstrap.servers': broker_url,
        'group.id': group_id,
        'auto.offset.reset': 'earliest'  # Start reading from earliest available message
    })

    # Subscribe to the topic
    consumer.subscribe([topic_name])

    print(f"Consuming messages from topic '{topic_name}'...")

    try:
        while True:
            # Poll for messages from Kafka
            message = consumer.poll(1.0)  # Wait up to 1 second for a message
            
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())
            else:
                # Successfully received a message
                print(f"Received message: {message.value().decode('utf-8')}")

    except KeyboardInterrupt:
        print("Consumption interrupted by user.")

    finally:
        # Close down the consumer to commit final offsets and clean up
        consumer.close()


if __name__ == "__main__":
    broker_url = 'localhost:9092'  
    # topic_name = 'test-topic'    
    topic_name = 'from_mysql_customers'    
    group_id = 'test-consumer-group'  

    # Start consuming messages
    consume_messages(broker_url, topic_name, group_id)
