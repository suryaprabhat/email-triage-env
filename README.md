---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Meta OpenEnv Hackathon 2024 - Email Triage Environment

## Overview
This repository contains a **production-ready email triage environment** for the Meta OpenEnv Hackathon. It implements a critical, real-world business task: categorizing and prioritizing incoming emails. This process is essential for customer support, IT operations, and security teams.

The environment challenges agents to classify emails accurately while considering the severity of the problem and penalizing misses on high-risk categories like spam/security alerts.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from email_triage_env import EmailTriageEnv; print('✓ Ready')"
```

### 2. Test Environment

```python
from email_triage_env import EmailTriageEnv

env = EmailTriageEnv(num_emails=5, seed=42)
obs, info = env.reset()

print(f"Subject: {obs.subject}")

action = {
    "email_id": obs.email_id,
    "category": "support",
    "severity": "high"
}
obs, reward, done, _, info = env.step(action)
print(f"Reward: {reward}, Score: {info['current_score']}")
```

### 3. Run Baseline

We provide a baseline implementation using an LLM.

```bash
export HF_TOKEN="sk-your-openai-key-here"
python baseline_inference.py --num-emails 5 --seed 42
```

### 4. Docker Deployment

```bash
docker build -t email-triage-env .
docker run -e HF_TOKEN="sk-..." email-triage-env
```

## API Reference

The environment adheres to the Gymnasium interface and uses `Pydantic` models for strong typing, in accordance with the OpenEnv specification.

### Observation Space
```python
class Observation(BaseModel):
    email_id: int
    sender: str
    subject: str
    body: str
    timestamp: str
    processed: bool
    current_category: Optional[str]
    current_severity: Optional[str]
    emails_processed: int
    total_emails: int
    score: float
```

### Action Space
```python
class Action(BaseModel):
    email_id: int
    category: str # [spam, support, billing, product, security, other]
    severity: str # [urgent, high, medium, low]
    notes: Optional[str]
```

## Difficulty Levels

| Level | Difficulty | Description | Expected Baseline Accuracy |
|-------|------------|-------------|----------------------------|
| Easy | 0.3 | 5 emails with clear intent | 85%+ |
| Medium| 0.6 | 10 emails, mixed content | 70-75% |
| Hard | 0.9 | 20 emails, adversarial elements| 65-70% |

## Grading Criteria
The overall score returned by the agent combines:
- Accuracy (+0.5 for Category, +0.5 for Severity)
- Speed / Efficiency bonus
- Safety penalty (Missing `spam` gives a heavy penalty of -0.3)

## Troubleshooting
- **Missing API Key**: The baseline script falls back to a deterministic mock if the key is missing or invalid. Set `HF_TOKEN` with a valid OpenAI key for actual GPT-4 inference.
- **Gymnasium Errors**: Please check that `gymnasium>=0.28.0` is properly installed via requirements.txt.

## Hackathon Submission Checklist

- [x] Environment runs without errors
- [x] OpenEnv spec is valid
- [x] Reward function is logical and real-world aligned
- [x] Includes baseline evaluation
- [x] Ready for Hugging Face Spaces deployment
