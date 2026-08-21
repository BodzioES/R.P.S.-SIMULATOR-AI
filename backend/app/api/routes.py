from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..ai.policies import LearnedPolicy, RandomPolicy
from ..sim.state import manager

router = APIRouter(prefix="/api")

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


@router.get("/state")
def get_state():
    env = manager.env
    if env is None:
        return {"running": False, "env": None}
    return {
        "running": manager.running,
        "step": env.steps,
        "populations": {t.name: c for t, c in env.populations.items()},
        "winner": env.winning_type.name if env.winning_type else None,
        "policy": manager.policy_name,
    }


@router.post("/sim/start")
async def start(mode: str = "random", model: str = "best"):
    if mode == "trained":
        model_path = CHECKPOINTS_DIR / model / "model.zip"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        policy = LearnedPolicy(str(model_path))
        manager.policy_name = f"AI ({model})"
    else:
        policy = RandomPolicy()
        manager.policy_name = "Random"
    manager.start(policy)
    return {"running": True, "policy": manager.policy_name}


@router.post("/sim/stop")
async def stop():
    manager.stop()
    return {"running": False}


@router.post("/sim/reset")
async def reset():
    manager.reset()
    return {"ok": True}


@router.get("/models")
def list_models():
    if not CHECKPOINTS_DIR.exists():
        return {"models": []}
    models = []
    for p in sorted(CHECKPOINTS_DIR.iterdir()):
        if p.is_dir() and (p / "model.zip").exists():
            models.append({"name": p.name, "path": str(p / "model.zip")})
    best = CHECKPOINTS_DIR / "best" / "model.zip"
    return {
        "models": models,
        "best_available": best.exists(),
    }
