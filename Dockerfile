FROM python:3.11-slim

WORKDIR /app

# Install only the API dependencies (anthropic is inference-only, not needed here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY data/        data/
COPY environment/ environment/
COPY api.py       api.py
COPY inference.py inference.py
COPY openenv.yaml openenv.yaml
COPY pyproject.toml pyproject.toml

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Use uvicorn directly — more reliable than python api.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
