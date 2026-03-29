FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY data/       data/
COPY environment/ environment/
COPY api.py      .
COPY inference.py .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

CMD ["python", "api.py"]
