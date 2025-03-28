from confluent_kafka import Producer

# Kafka configuration
conf = {
    'bootstrap.servers': 'localhost:54785'  # Update if using a different broker address
}

# Create a Producer instance
producer = Producer(conf)

# Callback for delivery reports
def delivery_report(err, msg):
    if err:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

# Produce messages to a topic
topic = 'test-topic'  # Ensure this topic exists in your Kafka instance
messages = ['Hello Kafka', 'This is a test', 'Kafka with Python is cool!']

for message in messages:
    producer.produce(topic, value=message, callback=delivery_report)
    producer.flush()  # Ensure message is sent

print("All messages produced successfully!")