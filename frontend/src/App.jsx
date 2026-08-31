import { useEffect, useState } from "react";
import Board from "./components/Board.jsx";
import Controls from "./components/Controls.jsx";
import PopulationChart from "./components/PopulationChart.jsx";
import { useWebSocket } from "./hooks/useWebSocket.js";

export default function App() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const { latest, historyRef, connected } = useWebSocket(
    `${proto}//${window.location.host}/ws`
  );

  const [models, setModels] = useState([]);
  const [boardSize, setBoardSize] = useState(8);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((data) => {
        setModels(data.models || []);
      })
      .catch(() => {});
  }, []);

  const call = (path, method = "POST") => {
    return fetch(path, { method });
  };

  const startRandom = () =>
    call(`/api/sim/start?mode=random&board_size=${boardSize}&agents_per_type=5&episode_length=200`);

  const startTrained = (model) =>
    call(`/api/sim/start?mode=trained&model=${model}&board_size=${boardSize}&agents_per_type=5&episode_length=200`);

  return (
    <div className="app">
      <h1>RPS Simulator AI</h1>
      <Controls
        connected={connected}
        policy={latest?.policy}
        models={models}
        boardSize={boardSize}
        onBoardSizeChange={setBoardSize}
        onStartRandom={startRandom}
        onStartTrained={startTrained}
        onStop={() => call("/api/sim/stop")}
        onReset={() => call("/api/sim/reset")}
      />
      <div className="layout">
        <Board snapshot={latest} boardSize={boardSize} />
        <div className="side">
          <h2>Populations</h2>
          <PopulationChart history={historyRef.current} />
          <div className="info">
            <p>step: {latest?.step ?? 0}</p>
            <p>winner: {latest?.winner ?? "-"}</p>
            <p>policy: {latest?.policy ?? "-"}</p>
            <p>
              populations:{" "}
              {latest ? JSON.stringify(latest.populations) : "-"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}