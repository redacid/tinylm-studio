import { useEffect, useMemo, useState } from "react";
import { EngineClient } from "./api";
import { Chat } from "./Chat";
import { Data } from "./Data";
import { Datasets } from "./Datasets";
import { Query } from "./Query";
import { EvaluationScreen } from "./Evaluation";
import { loadChatSettings, saveChatSettings } from "./chatSettings";
import { Models } from "./Models";
import { Settings } from "./Settings";
import { Training } from "./Training";
import { TabActive } from "./tabActive";
import type { EngineState } from "./types";
import { Icon, type IconName } from "./Icon";

const PHASE_LABELS: Record<EngineState["phase"], string> = {
  "setting-up": "Setting up the Python environment…",
  starting: "Starting engine…",
  ready: "Engine ready",
  error: "Engine error",
  stopped: "Engine stopped",
};

const PHASE_ICONS: Record<EngineState["phase"], { name: IconName; spin?: boolean }> = {
  "setting-up": { name: "loader-2", spin: true },
  starting: { name: "loader-2", spin: true },
  ready: { name: "circle-check" },
  error: { name: "alert-circle" },
  stopped: { name: "circle-off" },
};

type Tab = "chat" | "data" | "sql" | "datasets" | "training" | "evaluation" | "models" | "settings";

// Tabs that need the engine and keep their state while hidden (the chat has its own).
const PAGES: Tab[] = ["data", "sql", "datasets", "training", "evaluation", "models"];

const TAB_LABELS: Record<Tab, string> = {
  chat: "Chat",
  data: "Data",
  sql: "SQL",
  datasets: "Datasets",
  training: "Training",
  evaluation: "Evaluation",
  models: "Models",
  settings: "Settings",
};

const TAB_ICONS: Record<Tab, IconName> = {
  chat: "message",
  data: "database",
  sql: "sql",
  datasets: "table",
  training: "barbell",
  evaluation: "chart-bar",
  models: "cube",
  settings: "settings",
};

export function App() {
  const [engine, setEngine] = useState<EngineState>({ phase: "starting", where: "" });
  const [tab, setTab] = useState<Tab>("chat");
  const [visited, setVisited] = useState<Set<Tab>>(() => new Set([tab]));
  if (!visited.has(tab)) setVisited(new Set(visited).add(tab));
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
    () => (engine.phase === "ready" ? new EngineClient(engine.url, engine.apiKey) : null),
    [engine],
  );

  return (
    <div className="app">
      <header>
        <strong>tinylm-studio</strong>
        <nav>
          {(Object.keys(TAB_LABELS) as Tab[]).map((t) => (
            <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
              <Icon name={TAB_ICONS[t]} />
              {TAB_LABELS[t]}
            </button>
          ))}
        </nav>
        <span className={`status ${engine.phase}`}>
          <Icon {...PHASE_ICONS[engine.phase]} />
          {PHASE_LABELS[engine.phase]}
          {engine.phase === "starting" && engine.where && ` · ${engine.where}`}
          {engine.phase === "starting" && engine.detail && ` · ${engine.detail}`}
          {engine.phase === "ready" &&
            ` · v${engine.version}${engine.commit ? ` (${engine.commit})` : ""} · ${engine.where}`}
          {engine.phase === "error" && ` · ${engine.message}`}
          {engine.phase === "stopped" && engine.message && ` · ${engine.message}`}
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
        {tab === "settings" && (
          // Needs no engine: these settings live in the app.
          <Settings client={client} engine={engine} chat={chatSettings} onChat={changeChatSettings} />
        )}
        {!client && tab !== "settings" && tab !== "chat" && (
          <p className="muted">
            {engine.phase === "error" || engine.phase === "stopped"
              ? "No engine: see Settings › Engine."
              : "Waiting for the engine…"}
          </p>
        )}
        {/* The other tabs stay mounted once opened, so a form, a selection or the progress of a
            generation, crawl or training run is still there after switching tabs. */}
        {client &&
          PAGES.filter((page) => visited.has(page)).map((page) => (
            <div key={page} className="tab-page" hidden={tab !== page}>
              <TabActive.Provider value={tab === page}>
                {page === "data" ? (
                  <Data client={client} />
                ) : page === "sql" ? (
                  <Query client={client} />
                ) : page === "datasets" ? (
                  <Datasets client={client} />
                ) : page === "training" ? (
                  <Training client={client} onNeedModel={() => setTab("models")} />
                ) : page === "evaluation" ? (
                  <EvaluationScreen client={client} />
                ) : (
                  <Models client={client} />
                )}
              </TabActive.Provider>
            </div>
          ))}
      </main>
    </div>
  );
}
