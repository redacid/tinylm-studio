import { ReactNode, useState } from "react";
import { EngineClient } from "./api";
import { CHAT_DEFAULTS, ChatSettings } from "./chatSettings";
import { CrawlerSettingsForm } from "./CrawlerSettings";
import { EngineSettingsForm } from "./EngineSettings";
import { Logs } from "./Logs";
import { SearchSettingsForm } from "./SearchSettings";
import type { EngineState } from "./types";

type Section = "engine" | "chat" | "search" | "crawler" | "logs";

const SECTIONS: Record<Section, string> = {
  engine: "Engine",
  chat: "Chat settings",
  search: "Search settings",
  crawler: "Crawler settings",
  logs: "Logs",
};

/** App settings, saved as they are changed. Engine settings stay in .env. */
export function Settings({
  client,
  engine,
  chat,
  onChat,
}: {
  engine: EngineState;
  /** None while the engine starts; the engine keeps the search and crawler settings. */
  client: EngineClient | null;
  chat: ChatSettings;
  onChat: (settings: ChatSettings) => void;
}) {
  const [section, setSection] = useState<Section>("chat");
  return (
    <div className="data settings">
      <aside>
        {(Object.keys(SECTIONS) as Section[]).map((s) => (
          <button key={s} className={s === section ? "active" : ""} onClick={() => setSection(s)}>
            {SECTIONS[s]}
          </button>
        ))}
        <p className="muted">
          Engine and chat settings are kept by this app; search and crawler settings by the engine. Other engine settings (proxy,
          hardware, models folder) are in <code>.env</code>; see the user guide.
        </p>
      </aside>
      <section className="settings-page">
        {section === "engine" && <EngineSettingsForm engine={engine} />}
        {section === "chat" && <ChatSettingsForm value={chat} onChange={onChat} />}
        {section === "search" && <SearchSettingsForm client={client} />}
        {section === "crawler" && <CrawlerSettingsForm client={client} />}
        {section === "logs" && <Logs client={client} engine={engine} />}
      </section>
    </div>
  );
}

function ChatSettingsForm({ value, onChange }: { value: ChatSettings; onChange: (s: ChatSettings) => void }) {
  const set = (change: Partial<ChatSettings>) => onChange({ ...value, ...change });
  return (
    <div className="settings-form">
      <h3>Chat settings</h3>
      <p className="muted">Saved as you change them and kept after a restart.</p>

      <Field
        label="Default system prompt"
        hint="A new chat starts with it; the field above the chat can still change it."
      >
        <textarea
          rows={3}
          value={value.systemPrompt}
          placeholder="e.g. You are a helpful game guide. Answer briefly."
          onChange={(e) => set({ systemPrompt: e.target.value })}
        />
      </Field>

      <h4>Answers</h4>
      <div className="settings-grid">
        <Field
          label="Max answer length"
          hint="In tokens; a token is about ¾ of an English word, less in Ukrainian or Russian. Empty: the model's setting (Models tab), 512 by default."
        >
          <NumberInput
            value={value.maxTokens}
            min={16}
            max={8192}
            step={16}
            optional
            placeholder="model's"
            onChange={(n) => set({ maxTokens: n })}
          />
        </Field>
        <Field
          label="Temperature"
          hint="0 always picks the most likely word: the same answer every time. Higher is more varied. Empty: the model's setting, 0.7 by default."
        >
          <NumberInput
            value={value.temperature}
            min={0}
            max={2}
            step={0.05}
            optional
            placeholder="model's"
            onChange={(n) => set({ temperature: n })}
          />
        </Field>
        <Field
          label="Top P"
          hint="Picks only from the most likely words that together make up this share. 1 turns it off. Empty: the model's setting, 1 by default."
        >
          <NumberInput
            value={value.topP}
            min={0.05}
            max={1}
            step={0.05}
            optional
            placeholder="model's"
            onChange={(n) => set({ topP: n })}
          />
        </Field>
        <Field
          label="Repetition penalty"
          hint="Above 1 makes repeating words less likely; helps small models that loop. Empty: the model's setting, then the engine's (1.1 by default)."
        >
          <NumberInput
            value={value.repetitionPenalty}
            min={0.5}
            max={2}
            step={0.05}
            optional
            onChange={(n) => set({ repetitionPenalty: n })}
          />
        </Field>
        <Field
          label="Sources from data"
          hint="How many pieces of the chosen data an answer gets. More can answer more, but the prompt is longer and slower. Empty: the engine's setting (4 by default)."
        >
          <NumberInput value={value.sources} min={1} max={20} step={1} optional onChange={(n) => set({ sources: n })} />
        </Field>
      </div>
      <Field label="Stop sequences" hint="One per line: the answer ends before any of them.">
        <textarea rows={2} value={value.stop.join("\n")} onChange={(e) => set({ stop: e.target.value.split("\n") })} />
      </Field>

      <h4>Chat screen</h4>
      <Check checked={value.stream} onChange={(stream) => set({ stream })}>
        Show the answer as it is written (off: it appears when complete)
      </Check>
      <Check checked={value.showJson} onChange={(showJson) => set({ showJson })}>
        Show the <code>{"{ }"}</code> button with the JSON sent and received
      </Check>
      <Check checked={value.rememberChoice} onChange={(rememberChoice) => set({ rememberChoice })}>
        Start with the model and data chosen last time
      </Check>
      <Check checked={value.keepConversation} onChange={(keepConversation) => set({ keepConversation })}>
        Keep the conversation after the app is closed
      </Check>

      <div>
        <button onClick={() => onChange(CHAT_DEFAULTS)}>Reset to defaults</button>
      </div>
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <label className="settings-field">
      <span>{label}</span>
      {children}
      <small className="muted">{hint}</small>
    </label>
  );
}

export function Check({
  checked,
  onChange,
  children,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
}) {
  return (
    <label className="settings-check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{children}</span>
    </label>
  );
}

/** A number kept as typed; only valid values reach the settings. */
export function NumberInput({
  value,
  min,
  max,
  step,
  optional,
  placeholder,
  onChange,
}: {
  value: number | null;
  min: number;
  max: number;
  step: number;
  optional?: boolean;
  /** Shown in an empty optional field: the value used then. */
  placeholder?: string;
  onChange: (n: number | null) => void;
}) {
  const [text, setText] = useState(value === null ? "" : String(value));
  const [shown, setShown] = useState(value);
  if (value !== shown) {
    // Changed from outside, e.g. Reset to defaults.
    setShown(value);
    setText(value === null ? "" : String(value));
  }
  const n = Number(text);
  const valid = text.trim() === "" ? !!optional : Number.isFinite(n) && n >= min && n <= max;
  return (
    <span className="settings-number">
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={text}
        placeholder={optional ? (placeholder ?? "default") : undefined}
        className={valid ? "" : "invalid"}
        onChange={(e) => {
          const raw = e.target.value;
          setText(raw);
          const next = raw.trim() === "" ? null : Number(raw);
          const ok = next === null ? !!optional : Number.isFinite(next) && next >= min && next <= max;
          if (ok) {
            setShown(next);
            onChange(next);
          }
        }}
      />
      {!valid && <small className="bad">{optional ? `Empty, or ${min}–${max}` : `${min}–${max}`}</small>}
    </span>
  );
}
