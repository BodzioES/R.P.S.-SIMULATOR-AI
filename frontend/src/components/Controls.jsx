export default function Controls({ connected, onStart, onStop, onReset }) {
  return (
    <div className="controls">
      <button onClick={onStart}>Start</button>
      <button onClick={onStop}>Stop</button>
      <button onClick={onReset}>Reset</button>
      <span className={connected ? "status ok" : "status bad"}>
        {connected ? "polaczono (WS)" : "brak polaczenia"}
      </span>
    </div>
  );
}