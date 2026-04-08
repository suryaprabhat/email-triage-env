from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
from email_triage_env import EmailTriageEnv
import os

app = FastAPI(title="Email Triage OpenEnv API")

# Initialize the environment
seed = int(os.environ.get("ENV_SEED", 42))
env = EmailTriageEnv(seed=seed)

# Action model for OpenEnv /step endpoint
class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.post("/reset")
def reset():
    """Reset the environment and return the initial observation."""
    obs, info = env.reset()
    obs_data = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    return {"observation": obs_data, "info": info}

@app.post("/step")
def step(req: StepRequest):
    """Take a step in the environment with the given action."""
    obs, reward, done, truncated, info = env.step(req.action)
    obs_data = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    return {
        "observation": obs_data,
        "reward": float(reward),
        "done": done,
        "truncated": truncated,
        "info": info
    }

@app.get("/")
def read_root():
    """Serve the customized landing page."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Environment API is active."}

@app.get("/{filename}.md")
def serve_markdown(filename: str):
    """Serve markdown documents linked from the landing page."""
    path = f"{filename}.md"
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Document not found."}

def start():
    """Entry point for the [project.scripts] server CLI command."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
