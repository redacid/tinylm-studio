import { useEffect, useState } from "react";
import { EngineClient, EngineOptions, EngineOptionsChanges } from "./api";
import { Field } from "./SearchSettings";
import { Icon } from "./Icon";

type Name = "proxy" | "llama_cpp_dir" | "threads" | "repetition_penalty" | "rag_top_k" | "embedding_device" | "log_level";
type Draft = Record<Name, string>;

const NAMES: Name[] = [
  "proxy",
  "llama_cpp_dir",
  "threads",
  "repetition_penalty",
  "rag_top_k",
  "embedding_device",
  "log_level",
];
const NUMBERS: Name[] = ["threads", "repetition_penalty", "rag_top_k"];

const text = (value: string | number | null) => (value === null ? "" : String(value));
const draftOf = (s: EngineOptions): Draft =>
  Object.fromEntries(NAMES.map((name) => [name, text(s[name])])) as Draft;

const TOKEN_SOURCES = {
  app: "saved here",
  environment: "from HF_TOKEN",
  login: "from hf auth login",
} as const;

/** Engine settings that otherwise come from TINYLM_* variables: the Hugging Face token, the
 * proxy, hardware, answers, the GGUF converter and the log level. Kept by the engine. */
export function EngineOptionsForm({ client }: { client: EngineClient | null }) {
  const [saved, setSaved] = useState<EngineOptions | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!client) return;
    client.engineOptions().then(
      (s) => {
        setSaved(s);
        setDraft(draftOf(s));
      },
      (e) => setError(String(e)),
    );
  }, [client]);

  if (!client) return <p className="muted">These settings are kept by the engine; waiting for it to start…</p>;
  if (!draft || !saved) return error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>;

  const set = (change: Partial<Draft>) => {
    setDraft({ ...draft, ...change });
    setNotice(null);
  };
  const original = draftOf(saved);
  const dirty = token !== "" || NAMES.some((name) => draft[name] !== original[name]);

  async function save(changes: EngineOptionsChanges, done: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const next = await client!.saveEngineOptions(changes);
      setSaved(next);
      setDraft(draftOf(next));
      setToken("");
      setNotice(done);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function saveDraft() {
    // Only what changed, so the other settings keep following the engine's TINYLM_* variables.
    const changes: Record<string, string | number | null> = {};
    for (const name of NAMES) {
      if (draft![name] === original[name]) continue;
      const value = draft![name].trim();
      // An empty number goes back to the default; an empty proxy or folder means none.
      changes[name] = NUMBERS.includes(name) ? (value === "" ? null : Number(value)) : value;
    }
    if (token.trim()) changes.hf_token = token.trim();
    save(changes as EngineOptionsChanges, "Saved; in effect now.");
  }

  /** The "Use the default" button of a setting saved here. */
  const reset = (name: Name | "hf_token") =>
    saved.saved.includes(name) && (
      <button
        disabled={busy}
        title="Forget the value saved here and use the engine's environment again"
        onClick={() => save({ [name]: null }, "Back to the default.")}
      >
        <Icon name="restore" />
        Use the default
      </button>
    );
  const fromEnvironment = (value: string) => (value === "" ? "none" : value);

  const tokenOff = saved.saved.includes("hf_token") && !saved.hf_token_set;

  return (
    <div className="settings-form">
      <h3>Engine options</h3>
      <p className="muted">
        Kept by the engine this app is connected to, so they also apply to a remote engine or a RunPod pod, and they win
        over its <code>TINYLM_*</code> variables and <code>HF_TOKEN</code>, which give the defaults. Changes take effect
        at once, without a restart.
      </p>

      <h4>Hugging Face</h4>
      <Field
        label="Access token"
        hint="For gated models (Llama, Gemma, …) and faster downloads; read access is enough. Accept each gated model's license on its Hugging Face page with the same account. Kept by the engine and never shown again."
      >
        <span className="key-row">
          <span className={`badge ${saved.hf_token_set ? "done" : ""}`}>
            {saved.hf_token_set
              ? `${saved.hf_token_hint} · ${TOKEN_SOURCES[saved.hf_token_source ?? "app"]}`
              : tokenOff
                ? "switched off"
                : "no token"}
          </span>
          <input
            type="password"
            autoComplete="off"
            placeholder={saved.hf_token_set ? "type to replace" : "paste the token: hf_…"}
            value={token}
            onChange={(e) => {
              setToken(e.target.value);
              setNotice(null);
            }}
          />
          {saved.hf_token_source === "app" && (
            <button disabled={busy} onClick={() => save({ hf_token: null }, "Token removed.")}>
              <Icon name="trash" />
              Remove
            </button>
          )}
          {(saved.hf_token_source === "environment" || saved.hf_token_source === "login") && (
            <button disabled={busy} onClick={() => save({ hf_token: "" }, "The engine sends no token now.")}>
              <Icon name="circle-off" />
              Don't use it
            </button>
          )}
          {tokenOff && reset("hf_token")}
        </span>
      </Field>
      {saved.hf_token_source === "app" && saved.defaults.hf_token_hint && (
        <p className="muted">Removing it goes back to the engine's own token ({saved.defaults.hf_token_hint}).</p>
      )}

      <h4>Network</h4>
      <Field
        label="Proxy"
        hint={`For crawling and web search: http://user:pass@host:port, socks5://host:port or socks5h://… (DNS through the proxy). Empty: none. Default: ${fromEnvironment(text(saved.defaults.proxy))}.`}
      >
        <span className="key-row">
          <input
            value={draft.proxy}
            placeholder="none"
            spellCheck={false}
            onChange={(e) => set({ proxy: e.target.value })}
          />
          {reset("proxy")}
        </span>
      </Field>

      <h4>Hardware and answers</h4>
      <div className="settings-grid">
        <Field
          label="CPU threads"
          hint={`For answers and GGUF export. Empty: ${saved.defaults.threads ?? "physical cores, at most 8"}. Applies from the next model loaded.`}
        >
          <span className="key-row">
            <input
              type="number"
              min={1}
              max={256}
              value={draft.threads}
              placeholder="auto"
              onChange={(e) => set({ threads: e.target.value })}
            />
            {reset("threads")}
          </span>
        </Field>
        <Field
          label="Repetition penalty"
          hint={`For models without their own (Models tab); above 1 keeps small models from looping. Default ${saved.defaults.repetition_penalty}.`}
        >
          <span className="key-row">
            <input
              type="number"
              min={0.5}
              max={2}
              step={0.05}
              value={draft.repetition_penalty}
              onChange={(e) => set({ repetition_penalty: e.target.value })}
            />
            {reset("repetition_penalty")}
          </span>
        </Field>
        <Field
          label="Sources from data"
          hint={`Pieces of the chosen data an answer gets, when the model and the chat set none. Default ${saved.defaults.rag_top_k}.`}
        >
          <span className="key-row">
            <input
              type="number"
              min={1}
              max={50}
              value={draft.rag_top_k}
              onChange={(e) => set({ rag_top_k: e.target.value })}
            />
            {reset("rag_top_k")}
          </span>
        </Field>
        <Field
          label="Embedding device"
          hint={`Where the model that indexes and searches data runs. auto: an NVIDIA GPU when there is one. Default ${saved.defaults.embedding_device}.`}
        >
          <span className="key-row">
            <select value={draft.embedding_device} onChange={(e) => set({ embedding_device: e.target.value })}>
              <option value="auto">auto</option>
              <option value="cuda">cuda (NVIDIA GPU)</option>
              <option value="cpu">cpu</option>
              <option value="mps">mps (Apple GPU)</option>
            </select>
            {reset("embedding_device")}
          </span>
        </Field>
      </div>

      <h4>Export and logs</h4>
      <Field
        label="llama.cpp folder"
        hint={`A llama.cpp checkout on the engine's computer, for exporting to GGUF offline. Empty: llama.cpp's converter is downloaded from PyPI once. Default: ${fromEnvironment(text(saved.defaults.llama_cpp_dir))}.`}
      >
        <span className="key-row">
          <input
            value={draft.llama_cpp_dir}
            placeholder="download the converter"
            spellCheck={false}
            onChange={(e) => set({ llama_cpp_dir: e.target.value })}
          />
          {reset("llama_cpp_dir")}
        </span>
      </Field>
      <Field
        label="Log level"
        hint={`What goes to the engine's console and log (Settings > Logs). DEBUG also shows every successful request. Default ${saved.defaults.log_level}.`}
      >
        <span className="key-row">
          <select value={draft.log_level} onChange={(e) => set({ log_level: e.target.value })}>
            {["DEBUG", "INFO", "WARNING", "ERROR"].map((level) => (
              <option key={level}>{level}</option>
            ))}
          </select>
          {reset("log_level")}
        </span>
      </Field>

      {error && <p className="error">{error}</p>}
      {notice && <p className="notice">{notice}</p>}
      <div className="buttons-row">
        <button onClick={saveDraft} disabled={busy || !dirty}>
          <Icon name="device-floppy" />
          Save
        </button>
        {dirty && (
          <button
            onClick={() => {
              setDraft(original);
              setToken("");
            }}
            disabled={busy}
          >
            <Icon name="arrow-back-up" />
            Discard changes
          </button>
        )}
      </div>
    </div>
  );
}
