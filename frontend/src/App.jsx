import Board from "./components/Board.jsx";
import Controls from "./components/Controls.jsx";
import PopulationChart from "./components/PopulationChart.jsx";
import StatsPanel from "./components/StatsPanel.jsx";
import { useWebSocket } from "./hooks/useWebSocket.js";

export default function App() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const { latest, historyRef, connected } = useWebSocket(
    `${proto}//${window.location.host}/ws`
  );

  const call = (path, method = "POST") => {
    return fetch(path, { method });
  };

  const startRandom = () =>
    call(`/api/sim/start?mode=random&board_size=8&agents_per_type=5&episode_length=0`);

  const startTrained = () =>
    call(`/api/sim/start?mode=trained&board_size=8&agents_per_type=5&episode_length=0`);

  return (
    <div className="app">
      <h1>RPS Simulator AI</h1>
      <Controls
        connected={connected}
        policy={latest?.policy}
        onStartRandom={startRandom}
        onStartTrained={startTrained}
        onStop={() => call("/api/sim/stop")}
        onReset={() => call("/api/sim/reset")}
      />
      <div className="layout">
        <Board snapshot={latest} boardSize={8} />
        <div className="side">
          <StatsPanel snapshot={latest} />
          <h2>Populations</h2>
          <PopulationChart history={historyRef.current} />
          <div className="info">
            <p><strong>policy</strong><span>{latest?.policy ?? "-"}</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}
