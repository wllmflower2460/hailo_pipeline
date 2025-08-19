# Hailo Docker Environment

This directory contains Docker setup for Hailo development and inference.

## Prerequisites

- Docker and Docker Compose installed
- User added to the `docker` group
- Hailo device connected (optional)

## Setup

1. Ensure Docker is installed and running:
   ```bash
   sudo apt install -y docker.io docker-compose
   sudo systemctl enable docker
   sudo systemctl start docker
   sudo usermod -aG docker $USER   # Requires logout/login to take effect
   ```

2. Log out and log back in for the group changes to take effect

3. Start the Docker environment:
   ```bash
   ./start-hailo-docker.sh
   ```

## Available Services

The Docker environment provides two services:

1. **hailo-dev** - Development environment with Jupyter notebook
   - Access Jupyter at: http://localhost:8888 (Token: hailo)
   - Access shell: `docker exec -it hailo-dog-training bash`

2. **hailo-inference** - Inference service with access to Hailo device
   - Access shell: `docker exec -it hailo-inference bash`
   - This container has direct access to the Hailo device if connected

## Installing Hailo Packages

After entering the container, run:
```bash
./hailo_integration/docker/setup-hailo-python.sh
```

This will install the Hailo Python packages and check for device connectivity.

## Stopping the Environment

To stop the Docker environment:
```bash
cd hailo_integration/docker
docker-compose down
```

## Data Persistence

The Docker containers mount the following directories:
- Project directory: `/app`
- Data directory: `/app/data`
- Results directory: `/app/results`

Any changes made in these directories will persist outside the containers.
