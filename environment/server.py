"""
FastAPI server for OpenEnv compliance and Hugging Face Spaces.
Interactive API docs are available at /docs.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from environment.corruptor import Corruptor
from environment.env import DataForgeEnv, SurgeonAction
from environment.schemas import HEALTHCARE_SCHEMA, SURGEON_TOOLS


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = None
env_lock = asyncio.Lock()


class ResetRequest(BaseModel):
    tier: int = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global env
    clean_data = pd.read_csv(os.path.join(ROOT_DIR, "data", "healthcare_clean.csv"))
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
    return {
        "name": "DataForge Arena",
        "version": "1.0.0",
        "openenv_compliant": True,
        "surgeon_tools": {k: v["name"] for k, v in SURGEON_TOOLS.items()},
        "corruption_tiers": 3,
        "reward_signals": [
            "accuracy_delta",
            "constraint_alignment",
            "schema_alignment",
            "outlier_targeting",
            "reasoning_quality",
            "parse_bonus",
            "anti_hack",
        ],
        "endpoints": ["/health", "/info", "/reset", "/step", "/metrics", "/docs"],
    }


@app.post("/reset")
async def reset(payload: ResetRequest | None = Body(default=None)):
    tier = payload.tier if payload is not None else 1
    if tier not in (1, 2, 3):
        raise HTTPException(status_code=422, detail="tier must be 1, 2, or 3")

    async with env_lock:
        env._corruptor.force_tier(tier)
        observation = env.reset()
        return observation.model_dump()


@app.post("/step")
async def step(action: SurgeonAction):
    async with env_lock:
        if env._state is None:
            raise HTTPException(status_code=400, detail="Call /reset first")
        try:
            observation, reward, done, info = env.step(action)
            return {
                "observation": observation.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
            }
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Step failed: {str(exc)[:200]}") from exc


@app.get("/metrics")
def get_metrics():
    """
    Returns committed training and evaluation evidence.
    Judges can verify all claims programmatically.
    """

    def _load(path: str):
        try:
            with open(os.path.join(ROOT_DIR, path), "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def _csv_summary(path: str):
        try:
            df = pd.read_csv(os.path.join(ROOT_DIR, path))
            shaped_series = df.get("shaped_reward_total")
            if shaped_series is None:
                shaped_series = df.get("policy_shaping")
            if shaped_series is None:
                shaped_series = pd.Series([0.0])
            return {
                "rows": len(df),
                "steps": int(df["step"].max()),
                "first_reward": float(df["total_reward"].iloc[0]),
                "final_reward": float(df["total_reward"].iloc[-1]),
                "best_reward": float(df["total_reward"].max()),
                "mean_reward": round(float(df["total_reward"].mean()), 4),
                "parse_success_mean": round(float(df["parse_success_rate"].mean()), 4),
                "shaped_reward_mean": round(float(shaped_series.mean()), 4),
            }
        except Exception:
            return None

    return {
        "grpo_eval": _load("eval/results.json"),
        "heuristic_eval": _load("eval/heuristic_results.json"),
        "training_log_summary": _csv_summary("logs/training_log.csv"),
        "environment": {
            "name": "dataforge-arena",
            "version": "1.0.0",
            "reward_range": [-5.0, 8.0],
            "corruption_tiers": 3,
            "tools": 8,
            "schemas": ["healthcare", "financial"],
        },
    }


if __name__ == "__main__":
    uvicorn.run("environment.server:app", host="0.0.0.0", port=7860, reload=False)
