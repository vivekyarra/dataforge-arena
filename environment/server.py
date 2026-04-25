"""
FastAPI server for OpenEnv compliance and HF Spaces.
Interactive API docs available at /docs (Swagger UI).
"""
import asyncio
import os
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.schemas import HEALTHCARE_SCHEMA, SURGEON_TOOLS


env = None
env_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_data = pd.read_csv(os.path.join(root_dir, "data", "healthcare_clean.csv"))
    env = DataForgeEnv(
        corruptor=Corruptor(),
        schema=HEALTHCARE_SCHEMA,
        clean_data=clean_data,
    )
    yield


app = FastAPI(
    title="DataForge Arena",
    version="1.0.0",
    description="Adversarial RL environment for data quality repair. OpenEnv compliant.",
    lifespan=lifespan,
)

# Enable CORS so browser-based demo clients can call the API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "difficulty": env._corruptor.difficulty if env else 0,
        "epoch": env._corruptor._epoch if env else 0,
        "rolling_avg_reward": round(env._corruptor._rolling_avg(), 4) if env else 0,
    }


@app.get("/info")
async def info():
    """Environment metadata useful for judges exploring the API."""
    return {
        "name": "DataForge Arena",
        "version": "1.0.0",
        "openenv_compliant": True,
        "surgeon_tools": {k: v["name"] for k, v in SURGEON_TOOLS.items()},
        "corruption_tiers": 3,
        "reward_signals": [
            "accuracy_delta",
            "tool_logic",
            "reasoning",
            "efficiency",
            "anti_hack",
        ],
        "endpoints": ["/health", "/info", "/reset", "/step", "/docs"],
    }


@app.post("/reset")
async def reset():
    async with env_lock:
        obs = env.reset()
        return obs.model_dump()


@app.post("/step")
async def step(action: SurgeonAction):
    async with env_lock:
        if env._state is None:
            raise HTTPException(400, "Call /reset first")
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }


if __name__ == "__main__":
    uvicorn.run("environment.server:app", host="0.0.0.0", port=7860, reload=False)
