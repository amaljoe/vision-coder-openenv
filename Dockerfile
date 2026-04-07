FROM python:3.11-slim

WORKDIR /app

# Install pip dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Let Playwright install Chromium + all its own system dependencies
RUN playwright install chromium --with-deps

# Copy source and install package
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV PORT=7860

CMD ["sh", "-c", "uvicorn openenv.server.app:app --host 0.0.0.0 --port ${PORT}"]
