import { useState } from "react";

export default function Controls({
  connected,
  policy,
  models,
  boardSize,
  onBoardSizeChange,
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
          {models.map((m) => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
          <option value="best">best</option>
        </select>
        <select value={boardSize} onChange={(e) => onBoardSizeChange(Number(e.target.value))}>
          <option value={4}>4x4</option>
          <option value={8}>8x8</option>
          <option value={16}>16x16</option>
          <option value={32}>32x32</option>
          <option value={64}>64x64</option>
        </select>
        <button onClick={onStop}>Stop</button>
        <button onClick={onReset}>Reset</button>
      </div>
      <div className="status-bar">
        <span className={connected ? "status ok" : "status bad"}>
          {connected ? "WS" : "OFF"}
        </span>
        <span className="policy-badge">{policy ?? "-"}</span>
      </div>
    </div>
  );
}