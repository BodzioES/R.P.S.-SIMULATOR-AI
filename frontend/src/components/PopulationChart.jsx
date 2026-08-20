import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function PopulationChart({ history }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={history}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="step" />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="ROCK" stroke="#9e9e9e" dot={false} />
        <Line type="monotone" dataKey="PAPER" stroke="#4f8ef7" dot={false} />
        <Line type="monotone" dataKey="SCISSORS" stroke="#e74c3c" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}