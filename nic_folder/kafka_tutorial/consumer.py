from confluent_kafka import Consumer

# Kafka configuration
conf = {
    'bootstrap.servers': 'localhost:54785',
    'group.id': 'test-group',  # Consumer group
    'auto.offset.reset': 'earliest'  # Start from the beginning of the topic
}

# Create a Consumer instance
consumer = Consumer(conf)

# Subscribe to the topic
topic = 'test-topic'
consumer.subscribe([topic])

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
except KeyboardInterrupt:
    print("Stopping consumer...")
finally:
    consumer.close()