import random

from app.env.entities import Type
from app.env.rps_env import RPSEnv


def describe(populations):
    return {t.name: c for t, c in populations.items()}


def main():
    env = RPSEnv(board_size=32, agents_per_type=10, episode_length=300, seed=42)
    env.reset()
    total_conversions = 0
    print("start:", describe(env.populations))

    for s in range(env.episode_length):
        actions = {a.id: random.randint(0, 8) for a in env.agents}
        _, rewards, done, info = env.step(actions)
        total_conversions += info["conversions"]
        if s in (0, 49, 99, 199, 299) or done:
            winner = info["winning_type"]
            winner_name = winner.name if winner else "-"
            avg_reward = sum(rewards.values()) / len(rewards)
            print(
                f"krok {s + 1}: {describe(info['populations'])}  "
                f"konwersje (skum.): {total_conversions}  "
                f"srednia nagroda: {avg_reward:.3f}  wygrana: {winner_name}"
            )
        if done:
            break


if __name__ == "__main__":
    main()