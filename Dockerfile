FROM python:3.11-slim

# Install system deps needed by Playwright/Chromium and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates \
    fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 \
    libatk1.0-0 libcups2 libdbus-1-3 libgdk-pixbuf2.0-0 libgtk-3-0 \
    libnspr4 libnss3 libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 \
    libxss1 libxtst6 xdg-utils libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser
RUN playwright install chromium --with-deps

# Copy source and install package
COPY . .
RUN pip install --no-cache-dir -e .

# Expose HF Spaces default port
EXPOSE 7860

ENV PORT=7860

CMD ["sh", "-c", "uvicorn openenv.server.app:app --host 0.0.0.0 --port ${PORT}"]
