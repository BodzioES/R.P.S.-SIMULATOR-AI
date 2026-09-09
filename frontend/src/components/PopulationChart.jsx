import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#1e1e24",
      border: "1px solid rgba(255,255,255,0.06)",
      borderRadius: 8,
      padding: "8px 12px",
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 12,
    }}>
      <div style={{ color: "#6b6b76", marginBottom: 4 }}>Step {label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name === "ROCK" ? "🪨" : p.name === "PAPER" ? "📄" : "✂️"} {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
};

export default function PopulationChart({ history }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={history}>
        <defs>
          <linearGradient id="gradRock" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#9e9e9e" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#9e9e9e" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gradPaper" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#60a5fa" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gradScissors" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f87171" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#f87171" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
        <XAxis dataKey="step" stroke="#6b6b76" fontSize={11} fontFamily="'JetBrains Mono', monospace" />
        <YAxis allowDecimals={false} stroke="#6b6b76" fontSize={11} fontFamily="'JetBrains Mono', monospace" />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}
          formatter={(value) => (
            <span style={{ color: "#a8a8b3" }}>
              {value === "ROCK" ? "🪨" : value === "PAPER" ? "📄" : "✂️"} {value}
            </span>
          )}
        />
        <Area
          type="monotone"
          dataKey="ROCK"
          stroke="#9e9e9e"
          fill="url(#gradRock)"
          dot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="PAPER"
          stroke="#60a5fa"
          fill="url(#gradPaper)"
          dot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="SCISSORS"
          stroke="#f87171"
          fill="url(#gradScissors)"
          dot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
