#!/bin/bash

echo "Testing dog detection within the container..."
podman exec -it hailo-dog-training python3 /app/hailo_integration/hailo_dog_detection.py --image /app/data/sample/sample_dog.jpg --threshold 0.4
