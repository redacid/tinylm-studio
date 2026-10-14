import { FormEvent, useEffect, useRef, useState } from "react";
import { ChatMessage, ChatParams, ChatTrace, EngineClient, Source, Timings } from "./api";
import { ChatSettings, loadChatSession, saveChatSession } from "./chatSettings";
import { LiveStatus, Progress, Stats } from "./Stats";

/** A question and its answer share the trace of the request that answered it. */
type Turn = ChatMessage & { sources?: Source[]; timings?: Timings; model?: string; trace?: ChatTrace };
type TraceView = { trace: ChatTrace; side: "sent" | "received" };

export function Chat({
  client,
  active,
  settings,
  onNeedModel,
}: {
  client: EngineClient;
  /** The chat tab is shown; hidden, the chat keeps its conversation. */
  active: boolean;
  settings: ChatSettings;
  onNeedModel: () => void;
}) {
  const [saved] = useState(loadChatSession);
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState((settings.rememberChoice && saved.model) || "");
  const [system, setSystem] = useState(
    settings.keepConversation && saved.system !== undefined ? saved.system : settings.systemPrompt,
  );
  const [collections, setCollections] = useState<string[]>([]);
  // The collection each model answers from by default (its settings on the Models tab).
  const [modelData, setModelData] = useState<Record<string, string>>({});
  const [collection, setCollection] = useState((settings.rememberChoice && saved.collection) || "");
  const [messages, setMessages] = useState<Turn[]>(
    settings.keepConversation ? ((saved.messages ?? []) as Turn[]) : [],
  );
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [traceView, setTraceView] = useState<TraceView | null>(null);
  const abort = useRef<AbortController | null>(null);

  // A new default system prompt from the settings replaces the current one.
  const firstRender = useRef(true);
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    setSystem(settings.systemPrompt);
  }, [settings.systemPrompt]);

  // Kept for the next start; not while an answer is being written.
  useEffect(() => {
    if (busy) return;
    const choice = settings.rememberChoice ? { model, collection } : {};
    const kept = settings.keepConversation ? { system, messages } : {};
    if (!saveChatSession({ ...choice, ...kept }) && settings.keepConversation) {
      // Too large with the JSON of every exchange: keep the conversation without it.
      saveChatSession({ ...choice, system, messages: messages.map(({ trace: _, ...m }) => m) });
    }
  }, [messages, model, collection, system, busy, settings.rememberChoice, settings.keepConversation]);

  // Reloaded whenever the tab is shown: models and collections may have changed on other tabs.
  useEffect(() => {
    if (!active) return;
    client.catalogInfo().then(
      (info) => setModelData(Object.fromEntries(info.models.map((m) => [m.id, m.settings.collection ?? ""]))),
      () => {},
    );
    Promise.all([client.installedModels(), client.defaultModel()]).then(([ids, preferred]) => {
      setModels(ids);
      setModel((current) =>
        ids.includes(current) ? current : (ids.includes(preferred) ? preferred : ids[0]) || "",
      );
    }, (e) => setError(String(e)));
    client.collections().then(
      (list) => {
        const names = list.map((c) => c.name);
        setCollections(names);
        setCollection((current) => (names.includes(current) ? current : ""));
      },
      () => {},
    );
  }, [client, active]);

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || !model || busy) return;
    const trace: ChatTrace = { request: {}, chunks: [] };
    const question: Turn = { role: "user", content: input.trim(), trace };
    const history: ChatMessage[] = [...messages, question].map(({ role, content }) => ({ role, content }));
    setMessages([...messages, question, { role: "assistant", content: "", model, trace }]);
    setInput("");
    setError(null);

    const controller = (abort.current = new AbortController());
    setBusy(true);
    const started = performance.now();
    setProgress({ phase: "preparing", started, searching: !!collection, tokens: 0 });
    const prompt = system.trim() ? [{ role: "system" as const, content: system.trim() }, ...history] : history;
    try {
      const updateLast = (change: (last: Turn) => Turn) =>
        setMessages((current) => [...current.slice(0, -1), change(current[current.length - 1])]);
      const onText = (text: string) => {
        updateLast((last) => ({ ...last, content: last.content + text }));
        setProgress((p) =>
          p && { ...p, phase: "generating", tokens: p.tokens + 1, firstToken: p.firstToken ?? performance.now() },
        );
      };
      const params: ChatParams = {
        max_tokens: settings.maxTokens ?? undefined,
        temperature: settings.temperature ?? undefined,
        top_p: settings.topP ?? undefined,
        stop: settings.stop.some(Boolean) ? settings.stop.filter(Boolean) : undefined,
        repetition_penalty: settings.repetitionPenalty ?? undefined,
        rag_top_k: collection && settings.sources ? settings.sources : undefined,
      };
      await client.chat(model, prompt, onText, {
        signal: controller.signal,
        trace,
        params,
        stream: settings.stream,
        // "" is sent too: no data, even for a model with its own collection.
        collection,
        onStart: () => setProgress((p) => p && { ...p, phase: "reading", prepared: performance.now() }),
        onSources: (sources) => updateLast((last) => ({ ...last, sources })),
        onTimings: (timings) => updateLast((last) => ({ ...last, timings })),
      });
    } catch (e) {
      if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e));
    } finally {
      abort.current = null;
      setBusy(false);
      setProgress(null);
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
        <select
          value={model}
          onChange={(e) => {
            setModel(e.target.value);
            // A model with its own collection brings it along; it can still be changed.
            if (modelData[e.target.value]) setCollection(modelData[e.target.value]);
          }}
        >
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
            {m.trace && settings.showJson && (
              <button
                className="trace-button"
                title={m.role === "user" ? "Show the JSON sent" : "Show the JSON received"}
                onClick={() => setTraceView({ trace: m.trace!, side: m.role === "user" ? "sent" : "received" })}
              >
                {"{ }"}
              </button>
            )}
            {m.content || (busy ? "…" : "")}
            {busy && i === messages.length - 1 && progress && <LiveStatus progress={progress} />}
            {m.timings && <Stats timings={m.timings} model={m.model} sources={m.sources?.length} />}
            {m.sources && m.sources.length > 0 && (
              <ol className="sources">
                {m.sources.map((s) => (
                  <li key={s.index} value={s.index} title={s.text}>
                    <SourceLink source={s} />
                  </li>
                ))}
              </ol>
            )}
          </div>
        ))}
        {error && <div className="message error">{error}</div>}
      </div>
      {traceView && (
        <TraceDialog
          view={traceView}
          onSide={(side) => setTraceView({ ...traceView, side })}
          onClose={() => setTraceView(null)}
        />
      )}
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

/** The JSON of one exchange with the engine: the request, and the streamed chunks received. */
function TraceDialog({
  view,
  onSide,
  onClose,
}: {
  view: TraceView;
  onSide: (side: TraceView["side"]) => void;
  onClose: () => void;
}) {
  const { trace, side } = view;
  const first = trace.chunks.find((c): c is { prompt?: unknown } => typeof c === "object" && c !== null);
  const sections: [string, unknown][] =
    side === "sent"
      ? [
          ["Sent to the engine: POST /v1/chat/completions", trace.request],
          ["As the model got it, with the sources added", first?.prompt],
        ]
      : trace.request.stream === false
        ? [[`Received: HTTP ${trace.status ?? "…"}`, trace.chunks[0]]]
        : [[`Received: HTTP ${trace.status ?? "…"}, ${trace.chunks.length} chunks`, trace.chunks]];
  const text = (value: unknown) => JSON.stringify(value, null, 2);
  const all = sections.filter(([, value]) => value !== undefined);

  useEffect(() => {
    const close = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  return (
    <div className="trace-backdrop" onClick={onClose}>
      <div className="trace-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="trace-header">
          <div className="tabs">
            <button className={side === "sent" ? "active" : ""} onClick={() => onSide("sent")}>Sent</button>
            <button className={side === "received" ? "active" : ""} onClick={() => onSide("received")}>Received</button>
          </div>
          <button onClick={() => navigator.clipboard.writeText(all.map(([, v]) => text(v)).join("\n\n"))}>Copy</button>
          <button onClick={onClose}>Close</button>
        </div>
        {all.map(([title, value]) => (
          <section key={title}>
            <h4>{title}</h4>
            <pre>{text(value)}</pre>
          </section>
        ))}
      </div>
    </div>
  );
}

/** A source's name, linked to its web page or file when the collection knows where it came from. */
function SourceLink({ source }: { source: Source }) {
  const href = sourceUrl(source.source);
  if (!href) return <>{source.document}</>;
  // The site, or the file name; the full address is in the tooltip.
  const where = href.startsWith("file:") ? source.source.split("/").pop() : new URL(href).host;
  return (
    <a href={href} target="_blank" rel="noreferrer" title={`${source.source}\n\n${source.text}`}>
      {source.document}
      {where !== source.document && <span className="source-where"> · {where}</span>}
    </a>
  );
}

function sourceUrl(source: string): string | null {
  if (/^https?:\/\//i.test(source)) return source;
  // Files added from the command line keep their full path; uploads only their name.
  if (source.startsWith("/")) return "file://" + source.split("/").map(encodeURIComponent).join("/");
  return null;
}
