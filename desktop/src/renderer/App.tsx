import { useEffect, useMemo, useState } from "react";
import { EngineClient } from "./api";
import { Chat } from "./Chat";
import { Data } from "./Data";
import { Datasets } from "./Datasets";
import { EvaluationScreen } from "./Evaluation";
import { loadChatSettings, saveChatSettings } from "./chatSettings";
import { Models } from "./Models";
import { Settings } from "./Settings";
import { Training } from "./Training";
import type { EngineState } from "./types";

const PHASE_LABELS: Record<EngineState["phase"], string> = {
  "setting-up": "Setting up the Python environment…",
  starting: "Starting engine…",
  ready: "Engine ready",
  error: "Engine error",
  stopped: "Engine stopped",
};

type Tab = "chat" | "data" | "datasets" | "training" | "evaluation" | "models" | "settings";

const TAB_LABELS: Record<Tab, string> = {
  chat: "Chat",
  data: "Data",
  datasets: "Datasets",
  training: "Training",
  evaluation: "Evaluation",
  models: "Models",
  settings: "Settings",
};

export function App() {
  const [engine, setEngine] = useState<EngineState>({ phase: "starting", port: 0 });
  const [tab, setTab] = useState<Tab>("chat");
  const [chatSettings, setChatSettings] = useState(loadChatSettings);
  const changeChatSettings = (settings: typeof chatSettings) => {
    setChatSettings(settings);
    saveChatSettings(settings);
  };

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
        {/* The chat stays mounted on the other tabs, so the conversation is kept. */}
        {client && (
          <div className="chat-tab" hidden={tab !== "chat"}>
            <Chat
              client={client}
              active={tab === "chat"}
              settings={chatSettings}
              onNeedModel={() => setTab("models")}
            />
          </div>
        )}
        {tab === "settings" ? (
          // Needs no engine: these settings live in the app.
          <Settings chat={chatSettings} onChat={changeChatSettings} />
        ) : !client ? (
          <p className="muted">Waiting for the engine…</p>
        ) : tab === "chat" ? null : tab === "data" ? (
          <Data client={client} />
        ) : tab === "datasets" ? (
          <Datasets client={client} />
        ) : tab === "training" ? (
          <Training client={client} onNeedModel={() => setTab("models")} />
        ) : tab === "evaluation" ? (
          <EvaluationScreen client={client} />
        ) : (
          <Models client={client} />
        )}
      </main>
    </div>
  );
}
