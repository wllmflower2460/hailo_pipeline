# Dockerfile for Hailo Development Environment
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
RUN ln -snf /usr/share/zoneinfo/UTC /etc/localtime && echo UTC > /etc/timezone

# Install essential packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    pkg-config \
    htop \
    nano \
    vim \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip

# Create a working directory
WORKDIR /app

# Create a non-root user to run the container
RUN useradd -m -s /bin/bash hailo
RUN mkdir -p /home/hailo/.cache && chown -R hailo:hailo /home/hailo

# Copy our simplified requirements file
COPY hailo_integration/docker/requirements.txt /app/

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install PyTorch CPU version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch CPU version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set ownership for the application directory
RUN chown -R hailo:hailo /app

# Switch to non-root user
USER hailo

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8

# Default command
CMD ["/bin/bash"]
