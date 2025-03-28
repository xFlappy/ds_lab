from confluent_kafka import Consumer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

# Kafka configuration
conf = {
    'bootstrap.servers': 'localhost:54785',  # Replace with your Kafka broker's address
    'group.id': 'test-group',  # Consumer group
    'auto.offset.reset': 'earliest'  # Start from the beginning of the topic
}

# Create Consumer instance
consumer = Consumer(conf)

# Topics to subscribe to
topics = ['test-topic', 'test-topic2', 'test-topic3']
consumer.subscribe(topics)

# Prepare list to collect messages
messages = []

# Function to handle the delivery report (for debugging)
def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

print(f"Listening for messages on topics {topics}...")

# Consume messages
try:
    while True:
        msg = consumer.poll(1.0)  # Wait for up to 1 second
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        
        # Prepare message data to save
        message_data = {
            'topic': msg.topic(),
            'key': msg.key().decode('utf-8') if msg.key() else None,
            'value': msg.value().decode('utf-8') if msg.value() else None,
            'timestamp': msg.timestamp()[1],  # Unix timestamp in milliseconds
            'headers': msg.headers()  # Headers, if any
        }
        messages.append(message_data)
        
        # Optional: Stop after collecting a certain number of messages (e.g., 100)
        if len(messages) >= 100:
            break

except KeyboardInterrupt:
    print("Stopping consumer...")

finally:
    print(f"Consumed a total of {len(messages)} messages.")
    consumer.close()

# Convert messages to a DataFrame
df = pd.DataFrame(messages)

# Convert DataFrame to Parquet format
table = pa.Table.from_pandas(df)
pq.write_table(table, 'kafka_messages.parquet')

print(f"Saved {len(messages)} messages to 'kafka_messages.parquet'.")