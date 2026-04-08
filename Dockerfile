FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables
ENV HF_TOKEN=""

# Run a self-test on container startup to verify everything is working
CMD ["python", "-c", "from email_triage_env import EmailTriageEnv; env = EmailTriageEnv(5); obs, _ = env.reset(); print('✓ Environment ready!')"]
