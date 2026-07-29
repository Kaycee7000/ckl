# Use an NVIDIA CUDA base image that supports Python
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Prevent interactive prompts (like timezone selection) during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.12, dev tools, and pip (distutils removed)
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.12 python3.12-dev curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy your source code and docs into the container
COPY src/ ./src/
COPY docs/ ./docs/

# Expose the port your SSE server runs on (e.g., 8000)
EXPOSE 8000

# Start the MCP SSE server
CMD ["python3.12", "-m", "src.mcp_interface.sse_server"]
