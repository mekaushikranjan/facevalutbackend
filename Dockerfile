FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libopenblas-dev \
    liblapack-dev \
    git \
    wget \
    unzip \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Set environment variables for dlib build
ENV DLIB_USE_CUDA=0
ENV CMAKE_BUILD_TYPE=Release
ENV CMAKE_POLICY_VERSION_MINIMUM=3.25
ENV CFLAGS="-O3 -march=native -mtune=native"
ENV CXXFLAGS="-O3 -march=native -mtune=native"
ENV MAKEFLAGS="-j$(nproc)"

# Install dlib with optimized build settings
RUN pip install --no-cache-dir dlib==19.24.2 --no-build-isolation --config-settings="--global-option=build_ext" --config-settings="--global-option=-DUSE_AVX_INSTRUCTIONS=ON" --config-settings="--global-option=-DUSE_SSE4_INSTRUCTIONS=ON"

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads faces

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:$PORT"] 