import { useEffect, useRef } from "react";

const TYPE_INFO = {
  ROCK: { color: "#9e9e9e", label: "R" },
  PAPER: { color: "#4f8ef7", label: "P" },
  SCISSORS: { color: "#e74c3c", label: "S" },
};

const AGENT_RADIUS = 0.25;

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
    const radiusPx = AGENT_RADIUS * cell;

    for (const agent of snapshot.agents) {
      const info = TYPE_INFO[agent.type];
      const cx = (agent.x / boardSize) * size;
      const cy = (agent.y / boardSize) * size;
      ctx.beginPath();
      ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2);
      ctx.fillStyle = info.color;
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [snapshot, boardSize]);

  return <canvas ref={canvasRef} width={700} height={700} className="board" />;
}