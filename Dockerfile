FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libpcl-dev \
    libeigen3-dev \
    libopencv-dev \
    libyaml-cpp-dev \
    libosmium2-dev \
    libexpat1-dev \
    libbz2-dev \
    zlib1g-dev \
    ros-humble-pcl-conversions \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-eigen \
    ros-humble-rviz2 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages not available via apt
RUN pip3 install --no-cache-dir shapely tqdm

# Create ROS2 workspace
RUN mkdir -p /ros2_ws/src

# Copy project into workspace
COPY . /ros2_ws/src/osm_bki/

# Build the workspace
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select osm_bki --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source workspace on login
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

WORKDIR /ros2_ws
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
