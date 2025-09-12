# Base image
FROM debian:bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3-venv python3-opencv \
    git wget curl unzip \
    python3-gi python3-gst-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-rtsp \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a venv
RUN python3.11 -m venv /opt/venv --system-site-packages

# Activate and install packages inside venv
RUN /opt/venv/bin/pip  install \
    "numpy==1.26.4" \
    "pillow==11.3.0" \
    "pybind11==3.0.1" \
    "requests"

# Copy tflite_runtime wheel and install
COPY prebuilt/tflite_runtime-2.16.1-cp311-cp311-linux_aarch64.whl /tmp/
RUN /opt/venv/bin/pip install /tmp/tflite_runtime-2.16.1-cp311-cp311-linux_aarch64.whl && \
    rm /tmp/tflite_runtime-2.16.1-cp311-cp311-linux_aarch64.whl

# Copy prebuilt libedgetpu debs and install
COPY prebuilt/*.deb /tmp/
RUN dpkg -i /tmp/*.deb || apt-get -f install -y && rm -rf /tmp/*.deb

# Create a models directory
RUN mkdir -p /models

RUN curl -L -o /models/efficientdet_lite0_320_ptq_edgetpu.tflite \
       https://raw.githubusercontent.com/google-coral/test_data/master/efficientdet_lite0_320_ptq_edgetpu.tflite && \
    curl -L -o /models/efficientdet_lite0_320_ptq.tflite \
       https://raw.githubusercontent.com/google-coral/test_data/master/efficientdet_lite0_320_ptq.tflite && \
    curl -L -o /models/coco_labels.txt \
       https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Default working directory
WORKDIR /workspace 

CMD ["/bin/bash"]

