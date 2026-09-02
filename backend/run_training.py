import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ai.train import train


def main():
    parser = argparse.ArgumentParser(description="RPS Simulator AI - Trening PPO")
    parser.add_argument("--episodes", type=int, default=500, help="Liczba epizodow treningowych")
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--agents-per-type", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Wznow trening z istniejacego modelu")
    args = parser.parse_args()

    total_timesteps = args.episodes * args.episode_length
    print(f"RPS AI - Trening PPO")
    print(f"  plansza: {args.board_size}x{args.board_size}")
    print(f"  agenci: {args.agents_per_type} x 3 = {args.agents_per_type * 3}")
    print(f"  epizod: {args.episode_length} krokow")
    print(f"  trening: {args.episodes} epizodow ({total_timesteps} krokow)")
    print(f"  ewaluacja: co {args.eval_every} epizodow")
    print(f"  seed: {args.seed}")
    print(f"  resume: {args.resume}")
    print()

    train(
        total_episodes=args.episodes,
        eval_every=args.eval_every,
        board_size=args.board_size,
        agents_per_type=args.agents_per_type,
        episode_length=args.episode_length,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()