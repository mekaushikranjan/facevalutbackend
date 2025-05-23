# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for dlib and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
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

# Install CMake using the official binary distribution
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh \
    && chmod +x cmake-3.25.1-linux-x86_64.sh \
    && ./cmake-3.25.1-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.25.1-linux-x86_64.sh

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements file
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

# Set environment variables
ENV MONGODB_URI="mongodb+srv://kaushik2003singh:Fg3yrUlzZPaH9R7y@complaintapp.shaxxqw.mongodb.net/facevault?retryWrites=true&w=majority&appName=ComplaintApp"
ENV MONGODB_DB_NAME="facevault"
ENV JWT_SECRET_KEY="947832367c17c5421fc1d718cf7e66b8f2d9ad653b5c2dcfc22bf6461b2e040a"
ENV JWT_ALGORITHM="HS256"
ENV ACCESS_TOKEN_EXPIRE_MINUTES="10080"
ENV SMTP_HOST="smtp.gmail.com"
ENV SMTP_PORT="587"
ENV SMTP_USER="mritunjaykaushik1803@gmail.com"
ENV SMTP_PASSWORD="wzgl pppg sezx nlno"
ENV SMTP_FROM="facevault@gmail.com"
ENV SMTP_FROM_NAME="FaceVault"

# Expose the port your app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 