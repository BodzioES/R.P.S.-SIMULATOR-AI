import Board from "./components/Board.jsx";
import Controls from "./components/Controls.jsx";
import PopulationChart from "./components/PopulationChart.jsx";
import { useWebSocket } from "./hooks/useWebSocket.js";

const BOARD_SIZE = 64;

export default function App() {
  const { latest, historyRef, connected } = useWebSocket(
    `ws://${window.location.host}/ws`
  );

  const call = async (path, method = "POST") => {
    await fetch(path, { method });
  };

  return (
    <div className="app">
      <h1>RPS Simulator AI</h1>
      <Controls
        connected={connected}
        onStart={() => call("/api/sim/start")}
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