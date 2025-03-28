from confluent_kafka import Consumer, TopicPartition

# Kafka configuration
conf = {
    'bootstrap.servers': 'localhost:54785', # Broker address (in this case localhost at port 54785)
    'group.id': 'test-group',  # Consumer group
    'auto.offset.reset': 'earliest'  # Start from the beginning of the topic
}

# Create a Consumer instance
consumer = Consumer(conf)

# Subscribe to the topic
topic = 'test-topic'
consumer.subscribe([topic])

# Manually assign the offset to the earliest position (this is to always start from the beginning)
partitions = consumer.assignment()
for partition in partitions:
    consumer.seek(TopicPartition(topic, partition.partition, offset=0))

print(f"Listening for messages on topic '{topic}'...")
try:
    while True:
        msg = consumer.poll(1.0)  # Wait for up to 1 second
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        print(f"Received message: {msg.value().decode('utf-8')}")
        consumer.commit()  # Manually commit the offset after processing the message
except KeyboardInterrupt:
    print("Stopping consumer...")
finally:
    consumer.close()