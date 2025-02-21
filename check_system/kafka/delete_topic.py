from kafka.admin import KafkaAdminClient, NewTopic

# Kafka broker address (replace with your broker URL)
bootstrap_servers = 'localhost:9092'

# Create Kafka admin client
admin_client = KafkaAdminClient(
    bootstrap_servers=bootstrap_servers,
    client_id='kafka_topic_deleter'
)

# List all topics
topics = admin_client.list_topics()

# Filter out internal topics like __consumer_offsets or others starting with '__'
topics_to_delete = [topic for topic in topics if not topic.startswith('__')]

if topics_to_delete:
    print(f"Deleting topics: {topics_to_delete}")
    
    # Delete the topics
    admin_client.delete_topics(topics=topics_to_delete)
    print("All topics deleted.")
else:
    print("No topics to delete.")

# Close the admin client connection
admin_client.close()
