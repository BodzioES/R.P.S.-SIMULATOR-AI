import asyncio
import json

from ..env.rps_env import RPSEnv


def build_snapshot(env, info):
    return {
        "step": env.steps,
        "agents": [
            {"id": a.id, "type": a.type.name, "x": a.x, "y": a.y}
            for a in env.agents
        ],
        "populations": {t.name: c for t, c in info["populations"].items()},
        "done": env.done,
        "winner": info["winning_type"].name if info["winning_type"] else None,
    }


class SimulationManager:
    def __init__(self, tick_interval=0.033, win_pause=1.0, seed=None):
        self.tick_interval = tick_interval
        self.win_pause = win_pause
        self.env = RPSEnv(seed=seed)
        self.env.reset()
        self.policy = None
        self.policy_name = "none"
        self.clients = set()
        self.running = False
        self.task = None

    def start(self, policy):
        if self.running:
            return
        if self.env.done:
            self.env.reset()
        self.policy = policy
        self.running = True
        self.task = asyncio.create_task(self._run())

    def stop(self):
        self.running = False

    def reset(self):
        if self.env is not None:
            self.env.reset()

    async def _run(self):
        while self.running:
            actions = self.policy.actions(self.env)
            _, rewards, done, info = self.env.step(actions)
            await self._broadcast(build_snapshot(self.env, info))
            if done:
                await asyncio.sleep(self.win_pause)
                if self.running:
                    self.env.reset()
                    self.env.done = False
            await asyncio.sleep(self.tick_interval)

    async def _broadcast(self, snapshot):
        if not self.clients:
            return
        message = json.dumps(snapshot)
        stale = []
        for websocket in list(self.clients):
            try:
                await websocket.send_text(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.clients.discard(websocket)