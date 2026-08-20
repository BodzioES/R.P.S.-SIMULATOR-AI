from fastapi import APIRouter

from ..ai.policies import RandomPolicy
from ..sim.state import manager

router = APIRouter(prefix="/api")


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
    }


@router.post("/sim/start")
async def start():
    manager.start(RandomPolicy())
    return {"running": True}


@router.post("/sim/stop")
async def stop():
    manager.stop()
    return {"running": False}


@router.post("/sim/reset")
async def reset():
    manager.reset()
    return {"ok": True}