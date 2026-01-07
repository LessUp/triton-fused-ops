# Triton Fused Operators Development Environment
# Base: NVIDIA PyTorch with CUDA and Triton pre-installed

FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="Triton Fused Ops Team"
LABEL description="Development environment for Triton Fused Operators"

# Set working directory
WORKDIR /workspace

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-xdist>=3.0.0 \
    hypothesis>=6.0.0 \
    black>=23.0.0 \
    ruff>=0.1.0 \
    mypy>=1.0.0 \
    pytest-cov>=4.0.0

# Copy project files
COPY . /workspace/

# Install the package in development mode
RUN pip install -e .

# Default command
CMD ["/bin/bash"]
