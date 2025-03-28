from confluent_kafka import Consumer, TopicPartition

# Kafka configuration
conf = {
    'bootstrap.servers': 'localhost:54785',
    'group.id': 'test-group',
    'auto.offset.reset': 'earliest',  # Ensure to start from the earliest available message
    'enable.auto.commit': False,  # Disable auto-commit of offsets
}

consumer = Consumer(conf)
topic = 'test-topic2'

# Assign the consumer to the topic and partition
consumer.subscribe([topic])

print(f"Listening for messages on topic '{topic}'...")

try:
    # Ensure we are assigned to partitions
    partitions = consumer.assignment()
    while not partitions:
        partitions = consumer.assignment()

    # Manually seek the earliest position for each partition
    for partition in partitions:
        consumer.seek(TopicPartition(topic, partition.partition, offset=0))  # Seek to the earliest offset (0)

    # Start consuming messages from the beginning of the topic
    while True:
        msg = consumer.poll(1.0)  # Wait for up to 1 second
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        print(f"Received message: {msg.value().decode('utf-8')}")
        print(f"Message key: {msg.key()}")
        print(f"Offset: {msg.offset()}")
        print(f"Timestamp: {msg.timestamp()[0]}")
        print(f"Partition: {msg.partition()}")
        print(f"Topic: {msg.topic()}")
        print(f"Headers: {msg.headers()}")


        consumer.commit()  # Commit the offset after processing each message
except KeyboardInterrupt:
    print("Stopping consumer...")
finally:
    consumer.close()