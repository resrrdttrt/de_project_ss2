from confluent_kafka import Producer

def delivery_report(err, msg):
    """ Callback function for message delivery reports """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


def produce_messages(broker_url, topic_name, messages):
    # Create a Kafka producer
    producer = Producer({'bootstrap.servers': broker_url})

    for message in messages:
        try:
            # Send message to the specified topic
            producer.produce(topic_name, value=message.encode('utf-8'), callback=delivery_report)

            # Flush to ensure the message is delivered
            producer.poll(0)
        except Exception as e:
            print(f"Failed to produce message: {e}")

    # Wait for all messages in the producer queue to be delivered
    producer.flush()


if __name__ == "__main__":
    broker_url = 'localhost:9092'  # Replace with your Kafka broker URL
    topic_name = 'test-topic'          # Replace with your topic name
    messages = ['Hello Kafka!', 'This is a test message.', 'Producing multiple messages.']  # Replace with your messages

    # Produce messages to the Kafka topic
    produce_messages(broker_url, topic_name, messages)
