"""Podglad nauczonego modelu — ascii + zapis do pliku.

Uruchom po treningu:
  .venv\\Scripts\\Activate.ps1
  cd backend
  python view_trained.py              # probuje best/best_model.zip
  python view_trained.py --model checkpoints/best/best_model.zip --board-size 8 --agents 5 --steps 200 --delay 0.15
  python view_trained.py --random     # baseline losowy

Wymaga VecNormalize jesli trening go uzyl — skrypt automatycznie szuka checkpoints/vecnorm_stats.pkl
"""
import argparse
import time
from pathlib import Path

from app.ai.policies import LearnedPolicy, RandomPolicy
from app.env.rps_env import RPSEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/best/best_model.zip")
    p.add_argument("--board-size", type=int, default=None)
    p.add_argument("--agents", type=int, default=None, help="agents_per_type")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--delay", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random", action="store_true", help="uzyj RandomPolicy zamiast modelu")
    return p.parse_args()


def main():
    args = parse_args()

    if args.random:
        print(f"[VIEW] RandomPolicy  board={args.board_size} agents_per_type={args.agents}")
        policy = RandomPolicy(seed=args.seed)
        env = RPSEnv(board_size=args.board_size or 8, agents_per_type=args.agents or 5, episode_length=args.steps, seed=args.seed)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            # fallback na druga nazwe
            alt = model_path.parent / "model.zip"
            if alt.exists():
                model_path = alt
            else:
                print(f"[VIEW] Brak modelu: {args.model} (szukano tez {alt})")
                print("       Uruchom najpierw: python run_training.py")
                return
        print(f"[VIEW] LearnedPolicy  model={model_path}  board={args.board_size} agents={args.agents}")
        policy = LearnedPolicy(str(model_path), board_size=args.board_size, agents_per_type=args.agents)
        # LearnedPolicy probuje wywnioskowac board/agents z vecnorm — jesli nie podano, uzyj domyslnych z modelu
        bs = args.board_size or getattr(policy, "board_size", None) or 8
        apt = args.agents or getattr(policy, "agents_per_type", None) or 5
        env = RPSEnv(board_size=bs, agents_per_type=apt, episode_length=args.steps, seed=args.seed)

    env.reset()
    total_conv = 0
    for step in range(args.steps):
        actions = policy.actions(env)
        _, rewards, done, info = env.step(actions)
        total_conv += info["conversions"]
        pops = info["populations"]
        winner = info["winning_type"]
        print(f"\n--- krok {step+1:3d}  pops={ {k.name: v for k,v in pops.items()} }  conv={info['conversions']:2d}  sum_conv={total_conv:3d}  winner={winner.name if winner else '-'}")
        print(env.render())
        if done:
            print(f"\n[VIEW] Koniec w kroku {step+1} — winner={winner}")
            break
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
