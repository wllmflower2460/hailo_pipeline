#!/bin/bash

echo "Running dog detection in the container..."
podman exec -it hailo-dog-training python3 /app/hailo_integration/docker/container_dog_detection.py --image /app/data/sample/sample_dog.jpg --threshold 0.4 --output /tmp/dog_detection_result.jpg

echo "Checking the output file..."
podman exec -it hailo-dog-training ls -la /tmp/dog_detection_result.jpg

echo "Copying the result from the container to the host..."
mkdir -p /home/wllmflower/projects/dog_training_pipeline/results/visualizations
podman cp hailo-dog-training:/tmp/dog_detection_result.jpg /home/wllmflower/projects/dog_training_pipeline/results/visualizations/

echo "Results saved to: /home/wllmflower/projects/dog_training_pipeline/results/visualizations/dog_detection_result.jpg"
