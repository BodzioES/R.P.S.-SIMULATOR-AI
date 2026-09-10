from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..ai.policies import LearnedPolicy, RandomPolicy
from ..sim.state import manager

router = APIRouter(prefix="/api")

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

# auto-wybor modelu na podstawie rozmiaru planszy
MODEL_BY_SIZE = {
    8: "rps-1.0",
    16: "rps-1.0",
    32: "best",
}


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
async def start(mode: str = "random", model: str = "auto", board_size: int = 8, agents_per_type: int = 5, episode_length: int = 200):
    # auto-wybor modelu na podstawie rozmiaru planszy
    if model == "auto":
        model = MODEL_BY_SIZE.get(board_size, "best")

    if mode == "trained":
        # EvalCallback zapisuje best_model.zip — obsluz model.zip i best_model.zip
        for fname in ("model.zip", "best_model.zip"):
            cand = CHECKPOINTS_DIR / model / fname
            if cand.exists():
                model_path = cand
                break
        else:
            raise HTTPException(status_code=404, detail=f"Model not found in {CHECKPOINTS_DIR / model}")

        # Rekonfiguruj env do rozmiarow z treningu
        manager.reconfigure(board_size, agents_per_type, episode_length)

        vecnorm = CHECKPOINTS_DIR / model / "vecnorm_stats.pkl"
        try:
            policy = LearnedPolicy(
                str(model_path),
                board_size=board_size,
                agents_per_type=agents_per_type,
                vecnorm_path=str(vecnorm) if vecnorm.exists() else None,
            )
        except Exception:
            policy = LearnedPolicy(str(model_path))
        manager.policy_name = f"AI ({model})"
        # sprawdz czy jest name.txt z nazwa treningu
        name_file = CHECKPOINTS_DIR / model / "name.txt"
        if name_file.exists():
            manager.policy_name = f"AI ({name_file.read_text(encoding='utf-8').strip()})"
    else:
        # Random — tez uzyj podanych parametrow
        manager.reconfigure(board_size, agents_per_type, episode_length)
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
            name_file = p / "name.txt"
            display_name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else p.name
            models.append({"name": p.name, "display_name": display_name, "path": str(p / "model.zip")})
    best = CHECKPOINTS_DIR / "best" / "model.zip"
    return {
        "models": models,
        "best_available": best.exists(),
    }
