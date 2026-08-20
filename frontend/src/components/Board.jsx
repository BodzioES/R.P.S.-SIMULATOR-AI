import { useEffect, useRef } from "react";

const TYPE_INFO = {
  ROCK: { color: "#9e9e9e", label: "R" },
  PAPER: { color: "#4f8ef7", label: "P" },
  SCISSORS: { color: "#e74c3c", label: "S" },
};

export default function Board({ snapshot, boardSize }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const size = canvas.width;
    ctx.clearRect(0, 0, size, size);

    if (!snapshot) return;
    const cell = size / boardSize;

    for (const agent of snapshot.agents) {
      const info = TYPE_INFO[agent.type];
      const cx = (agent.x + 0.5) * cell;
      const cy = (agent.y + 0.5) * cell;
      ctx.beginPath();
      ctx.arc(cx, cy, cell * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = info.color;
      ctx.fill();
    }
  }, [snapshot, boardSize]);

  return <canvas ref={canvasRef} width={700} height={700} className="board" />;
}