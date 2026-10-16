FROM python:3.11-slim-bookworm

# Use cpu for a smaller CPU-only image. Default wheels support CUDA and CPU.
ARG TORCH_BACKEND=cu124
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    GDA_IN_CONTAINER=1 \
    GDA_PROJECT_DIR=/workspace \
    HOME=/tmp/gda-home \
    XDG_CACHE_HOME=/workspace/.cache \
    TORCH_HOME=/workspace/.cache/torch \
    MPLCONFIGDIR=/tmp/gda-matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt requirements-simulation.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/${TORCH_BACKEND} && \
    python -m pip install -r requirements.txt -r requirements-simulation.txt && \
    python -m pip check
COPY . .
COPY run.sh /usr/local/bin/gda-run
RUN chmod +x /usr/local/bin/gda-run /workspace/run.sh
ENTRYPOINT ["/usr/local/bin/gda-run"]
CMD ["help"]
