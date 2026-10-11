import { useEffect, useState } from "react";
import { Timings } from "./api";

/** What the engine is doing for the answer being streamed, as far as the client can tell. */
export type Progress = {
  phase: "preparing" | "reading" | "generating";
  started: number;
  searching: boolean;
  prepared?: number;
  firstToken?: number;
  // Streamed text fragments; llama.cpp sends one per token, so this is close to the token count.
  tokens: number;
};

const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
// Rates over a few milliseconds (e.g. the echo test model) are meaningless.
const rate = (perSecond: number, ms: number) => (ms >= 100 ? ` (${perSecond} tok/s)` : "");

export function LiveStatus({ progress }: { progress: Progress }) {
  const [now, setNow] = useState(performance.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(performance.now()), 200);
    return () => clearInterval(timer);
  }, []);

  const { phase, started, prepared, firstToken, tokens, searching } = progress;
  let text: string;
  if (phase === "preparing") {
    text = searching ? "Loading the model and searching your data" : "Loading the model";
  } else if (phase === "reading") {
    text = `Reading the prompt · ${seconds(now - (prepared ?? started))}`;
  } else {
    const elapsed = now - (firstToken ?? now);
    const speed = elapsed > 500 ? ` · ${((tokens * 1000) / elapsed).toFixed(1)} tok/s` : "";
    text = `Generating · ${tokens} tokens${speed}`;
  }
  return <div className="stats live">{text} · total {seconds(now - started)}</div>;
}

export function Stats({ timings: t, model, sources }: { timings: Timings; model?: string; sources?: number }) {
  const parts = [
    model,
    t.load_ms >= 100 && `model load ${seconds(t.load_ms)}`,
    sources !== undefined && `search ${seconds(t.retrieval_ms)} (${sources} sources)`,
    `prompt ${t.prompt_n} tokens in ${seconds(t.prompt_ms)}${rate(t.prompt_per_second, t.prompt_ms)}`,
    `answer ${t.predicted_n} tokens in ${seconds(t.predicted_ms)}${rate(t.predicted_per_second, t.predicted_ms)}`,
    `total ${seconds(t.total_ms)}`,
  ].filter(Boolean);
  return <div className="stats">{parts.join(" · ")}</div>;
}
