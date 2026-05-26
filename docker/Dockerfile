# Graphagate — CPU training / inference / ONNX-export image.
#
# Build context is the repository root:
#     docker build -f docker/Dockerfile -t graphagate .
#
# torch is installed from PyPI; on linux/arm64 that resolves to a CPU-only
# aarch64 wheel. GPU/CUDA training and TensorRT compilation are out of scope for
# this image (they belong to the NVIDIA workstation described in the document).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build metadata + dependency manifests first (better layer caching).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e .

# Artifacts (checkpoint, ONNX, norm_stats) are written here; mount a volume to persist.
VOLUME ["/app/public"]

# Run any module: train (default), score, export_onnx.
#   docker run --rm graphagate                      # -> graphagate.train
#   docker run --rm graphagate graphagate.score --eval
#   docker run --rm graphagate graphagate.export_onnx
ENTRYPOINT ["python", "-m"]
CMD ["graphagate.train"]
