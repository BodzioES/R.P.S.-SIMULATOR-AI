import { useEffect, useRef, useState } from "react";

export function useWebSocket(url) {
  const [latest, setLatest] = useState(null);
  const [connected, setConnected] = useState(false);
  const historyRef = useRef([]);

  useEffect(() => {
    let ws = new WebSocket(url);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (event) => {
      const snapshot = JSON.parse(event.data);
      setLatest(snapshot);
      historyRef.current = [
        ...historyRef.current.slice(-299),
        { step: snapshot.step, ...snapshot.populations },
      ];
    };

    return () => ws.close();
  }, [url]);

  return { latest, historyRef, connected };
}