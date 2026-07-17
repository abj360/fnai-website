FROM python:3.11-slim

# System deps (opencv-python-headless needs libglib; ffmpeg helps with video assets)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as a non-root user (uid 1000).
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first for better layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# App code + static assets
COPY --chown=user . .

# /app must be writable so startup.py can drop downloaded weights/videos here
RUN chown -R user:user /app
USER user

# startup.py fetches model weights from Google Drive, then execs uvicorn on 7860
EXPOSE 7860
CMD ["python", "startup.py"]
