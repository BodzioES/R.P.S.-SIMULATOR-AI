import { useState } from "react";

export default function Controls({
  connected,
  policy,
  onStartRandom,
  onStartTrained,
  onStop,
  onReset,
}) {
  const [model, setModel] = useState("best");

  return (
    <div className="controls">
      <div className="control-group">
        <button onClick={onStartRandom} className="btn-random">
          Start (Random)
        </button>
        <button onClick={() => onStartTrained(model)} className="btn-ai">
          Start (AI)
        </button>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="best">best</option>
        </select>
        <button onClick={onStop}>Stop</button>
        <button onClick={onReset}>Reset</button>
      </div>
      <div className="status-bar">
        <span className={connected ? "status ok" : "status bad"}>
          {connected ? "WS" : "brak"}
        </span>
        <span className="policy-badge">{policy ?? "-"}</span>
      </div>
    </div>
  );
}