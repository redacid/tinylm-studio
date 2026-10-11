import { useEffect, useMemo, useState } from "react";
import { EngineClient } from "./api";
import { Chat } from "./Chat";
import { Data } from "./Data";
import { Models } from "./Models";
import type { EngineState } from "./types";

const PHASE_LABELS: Record<EngineState["phase"], string> = {
  "setting-up": "Setting up the Python environment…",
  starting: "Starting engine…",
  ready: "Engine ready",
  error: "Engine error",
  stopped: "Engine stopped",
};

type Tab = "chat" | "data" | "models";

const TAB_LABELS: Record<Tab, string> = { chat: "Chat", data: "Data", models: "Models" };

export function App() {
  const [engine, setEngine] = useState<EngineState>({ phase: "starting", port: 0 });
  const [tab, setTab] = useState<Tab>("chat");

  useEffect(() => {
    window.tinylm.onEngineState(setEngine);
    window.tinylm.getEngineState().then(setEngine);
  }, []);

  const client = useMemo(
    () => (engine.phase === "ready" ? new EngineClient(`http://127.0.0.1:${engine.port}`) : null),
    [engine],
  );

  return (
    <div className="app">
      <header>
        <strong>tinylm-studio</strong>
        <nav>
          {(Object.keys(TAB_LABELS) as Tab[]).map((t) => (
            <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
              {TAB_LABELS[t]}
            </button>
          ))}
        </nav>
        <span className={`status ${engine.phase}`}>
          {PHASE_LABELS[engine.phase]}
          {engine.phase === "ready" && ` · v${engine.version} · port ${engine.port}`}
          {engine.phase === "error" && ` · ${engine.message}`}
        </span>
      </header>
      <main>
        {!client ? (
          <p className="muted">Waiting for the engine…</p>
        ) : tab === "chat" ? (
          <Chat client={client} onNeedModel={() => setTab("models")} />
        ) : tab === "data" ? (
          <Data client={client} />
        ) : (
          <Models client={client} />
        )}
      </main>
    </div>
  );
}
