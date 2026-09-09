export default function StatsPanel({ snapshot }) {
  const step = snapshot?.step ?? 0;
  const pops = snapshot?.populations ?? {};
  const winner = snapshot?.winner;
  const rock = pops.ROCK ?? 0;
  const paper = pops.PAPER ?? 0;
  const scissors = pops.SCISSORS ?? 0;
  const total = rock + paper + scissors || 1;

  const maxPop = Math.max(rock, paper, scissors);
  const dominant =
    maxPop === rock ? "ROCK" :
    maxPop === paper ? "PAPER" : "SCISSORS";
  const dominantPct = ((maxPop / total) * 100).toFixed(0);

  return (
    <div className="stats-panel">
      <h2>Stats</h2>

      <div className="progress-section">
        <div className="progress-label">
          <span>Step</span>
          <span>{step} / 300</span>
        </div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${(step / 300) * 100}%` }}
          />
        </div>
      </div>

      <div className="faction-counts">
        <div className="faction-count">
          <span className="faction-emoji">🪨</span>
          <span className="faction-name">Rock</span>
          <span className="faction-number rock">{rock}</span>
        </div>
        <div className="faction-count">
          <span className="faction-emoji">📄</span>
          <span className="faction-name">Paper</span>
          <span className="faction-number paper">{paper}</span>
        </div>
        <div className="faction-count">
          <span className="faction-emoji">✂️</span>
          <span className="faction-name">Scissors</span>
          <span className="faction-number scissors">{scissors}</span>
        </div>
      </div>

      <div className="dominance">
        {winner ? (
          <strong>{winner} wins!</strong>
        ) : (
          <span>Dominant: <strong>{dominant}</strong> ({dominantPct}%)</span>
        )}
      </div>
    </div>
  );
}
