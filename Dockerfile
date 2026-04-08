FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables
ENV HF_TOKEN=""

# Run a self-test and keep the container alive serving files on port 7860 for Hugging Face
CMD ["sh", "-c", "python -c 'from email_triage_env import EmailTriageEnv; print(\"✓ Environment ready!\")' && python -m http.server 7860"]
