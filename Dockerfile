# 1. Base Image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies for WeasyPrint and Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# 4. Install uv
RUN pip install uv

# 5. Copy only dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock* ./

# 6. Install Python dependencies using uv
RUN uv pip install . --system

# 7. Copy the rest of the application code
COPY . .

# 8. Install playwright browsers and their dependencies
RUN playwright install --with-deps chromium

# 9. Expose the port the app runs on
EXPOSE 8000

# 10. Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 