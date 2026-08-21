import { useEffect, useState } from "react";
import Board from "./components/Board.jsx";
import Controls from "./components/Controls.jsx";
import PopulationChart from "./components/PopulationChart.jsx";
import { useWebSocket } from "./hooks/useWebSocket.js";

const BOARD_SIZE = 64;

export default function App() {
  const { latest, historyRef, connected } = useWebSocket(
    `ws://${window.location.host}/ws`
  );

  const [models, setModels] = useState([]);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((data) => {
        setModels(data.models || []);
      })
      .catch(() => {});
  }, []);

  const call = async (path, method = "POST") => {
    await fetch(path, { method });
  };

  return (
    <div className="app">
      <h1>RPS Simulator AI</h1>
      <Controls
        connected={connected}
        policy={latest?.policy}
        onStartRandom={() => call("/api/sim/start?mode=random")}
        onStartTrained={(model) =>
          call(`/api/sim/start?mode=trained&model=${model}`)
        }
        onStop={() => call("/api/sim/stop")}
        onReset={() => call("/api/sim/reset")}
      />
      <div className="layout">
        <Board snapshot={latest} boardSize={BOARD_SIZE} />
        <div className="side">
          <h2>Populacje</h2>
          <PopulationChart history={historyRef.current} />
          <div className="info">
            <p>krok: {latest?.step ?? 0}</p>
            <p>zwyciezca: {latest?.winner ?? "-"}</p>
            <p>polityka: {latest?.policy ?? "-"}</p>
            <p>
              populacje:{" "}
              {latest ? JSON.stringify(latest.populations) : "-"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}