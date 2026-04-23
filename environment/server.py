"""FastAPI server -- required for OpenEnv compliance and HF Spaces."""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import uvicorn

from environment.env import DataForgeEnv, SurgeonAction
from environment.corruptor import Corruptor
from environment.schemas import HEALTHCARE_SCHEMA

env = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    clean_data = pd.read_csv("data/healthcare_clean.csv")
    env = DataForgeEnv(
        corruptor=Corruptor(),
        schema=HEALTHCARE_SCHEMA,
        clean_data=clean_data,
    )
    yield

app = FastAPI(title="DataForge Arena", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "difficulty": env._corruptor.difficulty if env else 0,
        "epoch": env._corruptor._epoch if env else 0,
    }

@app.post("/reset")
async def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
async def step(action: SurgeonAction):
    if env._state is None:
        raise HTTPException(400, "Call /reset first")
    obs, reward_dict, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward_dict,
        "done": done,
        "info": info,
    }

if __name__ == "__main__":
    uvicorn.run("environment.server:app", host="0.0.0.0", port=7860, reload=False)
