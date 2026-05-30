# Graphagate — GPU training / verification image for the streaming TGN.
#
# Build context is the repository root:
#     docker build -f docker/Dockerfile -t graphagate .
#
# Uses CUDA 13 base image for RTX Blackwell support.
FROM nvidia/cuda:13.2.1-base-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Build metadata + dependency manifests first (better layer caching).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e .

# Artifacts (tgn_checkpoint.pt, tgn_stats.json) are written here; mount a volume to persist.
VOLUME ["/app/public"]

# Run any module: train_tgn (default), verify_tgn.
#   docker run --rm --gpus all -v "$PWD/public:/app/public" graphagate                       # -> graphagate.train_tgn
#   docker run --rm --gpus all -v "$PWD/public:/app/public" graphagate graphagate.verify_tgn
ENTRYPOINT ["python", "-m"]
CMD ["graphagate.train_tgn"]
