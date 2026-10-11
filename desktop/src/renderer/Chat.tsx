import { FormEvent, useEffect, useRef, useState } from "react";
import { ChatMessage, EngineClient, Source } from "./api";

type Turn = ChatMessage & { sources?: Source[] };

export function Chat({ client, onNeedModel }: { client: EngineClient; onNeedModel: () => void }) {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [system, setSystem] = useState("");
  const [collections, setCollections] = useState<string[]>([]);
  const [collection, setCollection] = useState("");
  const [messages, setMessages] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    client.installedModels().then((ids) => {
      setModels(ids);
      setModel((current) => current || ids[0] || "");
    }, (e) => setError(String(e)));
    client.collections().then((list) => setCollections(list.map((c) => c.name)), () => {});
  }, [client]);

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || !model || busy) return;
    const history: ChatMessage[] = [...messages, { role: "user" as const, content: input.trim() }].map(
      ({ role, content }) => ({ role, content }),
    );
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setError(null);

    const controller = (abort.current = new AbortController());
    setBusy(true);
    const prompt = system.trim() ? [{ role: "system" as const, content: system.trim() }, ...history] : history;
    try {
      const updateLast = (change: (last: Turn) => Turn) =>
        setMessages((current) => [...current.slice(0, -1), change(current[current.length - 1])]);
      await client.chat(model, prompt, (text) => updateLast((last) => ({ ...last, content: last.content + text })), {
        signal: controller.signal,
        collection: collection || undefined,
        onSources: (sources) => updateLast((last) => ({ ...last, sources })),
      });
    } catch (e) {
      if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e));
    } finally {
      abort.current = null;
      setBusy(false);
    }
  }

  if (models.length === 0) {
    return (
      <div className="empty">
        <p>No models installed yet.</p>
        <button onClick={onNeedModel}>Download a model</button>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="toolbar">
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((id) => <option key={id}>{id}</option>)}
        </select>
        <select value={collection} onChange={(e) => setCollection(e.target.value)} title="Answer from uploaded data">
          <option value="">No data</option>
          {collections.map((name) => <option key={name} value={name}>Data: {name}</option>)}
        </select>
        <input
          className="system"
          placeholder="System prompt (optional)"
          value={system}
          onChange={(e) => setSystem(e.target.value)}
        />
        <button onClick={() => setMessages([])} disabled={busy}>New chat</button>
      </div>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            {m.content || (busy ? "…" : "")}
            {m.sources && m.sources.length > 0 && (
              <ol className="sources">
                {m.sources.map((s) => (
                  <li key={s.index} value={s.index} title={s.text}>{s.document}</li>
                ))}
              </ol>
            )}
          </div>
        ))}
        {error && <div className="message error">{error}</div>}
      </div>
      <form className="composer" onSubmit={send}>
        <textarea
          value={input}
          placeholder="Ask something…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) send(e);
          }}
        />
        {busy ? (
          <button type="button" onClick={() => abort.current?.abort()}>Stop</button>
        ) : (
          <button type="submit" disabled={!input.trim()}>Send</button>
        )}
      </form>
    </div>
  );
}
