import { useEffect, useRef } from "react";

const TYPE_INFO = {
  ROCK: { color: "#9e9e9e", emoji: "🪨", glow: "rgba(158, 158, 158, 0.3)" },
  PAPER: { color: "#60a5fa", emoji: "📄", glow: "rgba(96, 165, 250, 0.3)" },
  SCISSORS: { color: "#f87171", emoji: "✂️", glow: "rgba(248, 113, 113, 0.3)" },
};

const AGENT_RADIUS = 0.5;

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
    const fontSize = Math.max(10, Math.floor(cell * 0.8));

    for (const agent of snapshot.agents) {
      const info = TYPE_INFO[agent.type];
      const cx = (agent.x / boardSize) * size;
      const cy = (agent.y / boardSize) * size;

      ctx.shadowColor = info.glow;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2);
      ctx.fillStyle = info.color + "33";
      ctx.fill();
      ctx.strokeStyle = info.color + "26";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.shadowBlur = 0;

      ctx.font = `${fontSize}px serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(info.emoji, cx, cy);
    }
  }, [snapshot, boardSize]);

  return <canvas ref={canvasRef} width={700} height={700} className="board" />;
}
