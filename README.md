# RPS Simulator AI — Rock-Paper-Scissors Multi-Agent + PPO

**Live demo:** https://rps-simulator.kuncrog.com/

![Stack](https://img.shields.io/badge/stack-FastAPI%20%7C%20React%20%7C%20PPO-blue) ![Python](https://img.shields.io/badge/python-3.12-blue) ![Docker](https://img.shields.io/badge/docker-compose-blue)

---

## 1. Stack & Versions

| Layer | Technology | Version |
|---------|-------------|--------|
| Backend | Python | 3.12 (slim) |
| API | FastAPI + Uvicorn | `fastapi`, `uvicorn[standard]` |
| RL | stable-baselines3 (PPO) | `>=2.3` |
| ML | PyTorch | `torch` |
| Env | Gymnasium | `gymnasium` |
| Logs | TensorBoard | `tensorboard` |
| Tests | pytest | `pytest` |
| Frontend | React + Vite | `react 18.3.1`, `vite 5.4.11` |
| Charts | Recharts | `2.13.3` |
| Prod frontend | Nginx | `nginx:alpine` |
| Frontend build | Node | `node:20-alpine` |
| Deploy | Docker Compose + GitHub Actions + Nginx + Certbot |  |

---

## 2. How the Simulation Works

A grid-based world where three populations compete — **Rock, Paper, Scissors**.

Each agent moves freely on the board. When two agents of different types meet on the same cell, the winner converts the loser to its own type — for example Rock converts Scissors, Scissors converts Paper, and Paper converts Rock.

The total number of agents never changes. The simulation ends when one type takes over the entire population or the step limit is reached. The interface shows live population charts and the board updates in real time via WebSocket.

The AI is trained with **PPO (Proximal Policy Optimization)**. Each agent learns its own movement policy — where to go based on what it sees around it. Rewards encourage converting opponents, pushing toward dominance, and winning quickly, with a penalty for draws.

The project is educational: it demonstrates multi-agent interaction, reinforcement learning, and real-time visualization in a simple, visual game.

---

## 3. Run Locally

### Requirements
- Python 3.12
- Node 20
- (optional) Docker Desktop

### Quick start (dev)

**Backend:**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
cd backend
python -m pytest tests -q
uvicorn app.main:app --reload --port 8000
```

**Frontend (new terminal):**
```powershell
cd frontend
npm ci
npm run dev
# open http://localhost:5173
```

**Docker (prod build):**
```powershell
docker compose up -d --build
# frontend → http://localhost:81
# backend  → http://localhost:8000/docs
```

### How to use (graphical)

1. Open the app in your browser.
2. Use **Controls** to `Start` (random or trained model), `Stop`, and `Reset` the simulation.
3. Watch the **Board** — colors represent Rock / Paper / Scissors.
4. Track **PopulationChart** and counters for steps, winner, and populations — they update live.

No config editing is needed for a first run. Screenshots can be added here if needed.

---

## 4. Try it Online

No install needed — open **https://rps-simulator.kuncrog.com/** and use the controls as described above. The hosted version runs the same Docker build as local.

