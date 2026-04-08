FROM python:3.11-slim

WORKDIR /app

# Install pip dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torch CPU-only (avoids ~2 GB of GPU kernels)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Pre-download CLIP model so first request isn't slow
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); print('CLIP ready')"

# Let Playwright install Chromium + all its own system dependencies
RUN playwright install chromium --with-deps

# Copy source and install package
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV PORT=7860

CMD ["sh", "-c", "uvicorn openenv.server.app:app --host 0.0.0.0 --port ${PORT}"]
