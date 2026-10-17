import {
  Fragment,
  ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { EngineClient } from "./api";
import type { EngineState } from "./types";
import { Icon, type IconName } from "./Icon";

type Source = "engine" | "app";
const LEVEL_ICONS: Record<Level, IconName> = {
  DEBUG: "bug",
  INFO: "info-circle",
  WARNING: "alert-triangle",
  ERROR: "alert-circle",
  CRITICAL: "flame",
};

type Level = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";
type Entry = {
  date: string;
  time: string;
  level: Level;
  scope: string;
  message: string;
  more: string;
};

const SHOWN = 1000;
const RANK: Record<Level, number> = {
  DEBUG: 0,
  INFO: 1,
  WARNING: 2,
  ERROR: 3,
  CRITICAL: 4,
};
// "2026-10-16 04:39:23.042 INFO engine.api.knowledge: Collection games created" (+ a traceback).
const LINE =
  /^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d\.\d{3}) (DEBUG|INFO|WARNING|ERROR|CRITICAL) ([^:\s]+): ([^\n]*)\n?([\s\S]*)$/;

// Short names for the server's own loggers.
function scopeName(scope: string): string {
  if (scope === "uvicorn.access") return "http";
  if (scope.startsWith("uvicorn")) return "server";
  if (scope === "py.warnings") return "warning";
  return scope.replace(/^engine\./, "");
}

function parse(line: string): Entry {
  const m = LINE.exec(line);
  if (!m)
    return {
      date: "",
      time: "",
      level: "INFO",
      scope: "",
      message: line,
      more: "",
    };
  return {
    date: m[1],
    time: m[2],
    level: m[3] as Level,
    scope: scopeName(m[4]),
    message: m[5],
    more: m[6],
  };
}

// A stable color per scope, from a small palette.
const SCOPE_COLORS = ["purple", "teal", "accent", "warn", "ok"];
function scopeColor(scope: string): string {
  if (scope === "http") return "teal";
  if (scope === "server" || scope === "app") return "muted";
  let hash = 0;
  for (const c of scope) hash = (hash * 31 + c.charCodeAt(0)) >>> 0;
  return SCOPE_COLORS[hash % SCOPE_COLORS.length];
}

// HTTP requests, numbers with units, quoted names and addresses stand out in a message.
const ACCESS =
  /^(\S+) - "(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) (\S+) HTTP\/[\d.]+" (\d{3})$/;
const TOKENS =
  /(https?:\/\/[^\s,;]+|\b\d+(?:\.\d+)? ?(?:ms|s|tok\/s|GB|MB|KB|chunks?|documents?|pages?|examples?|tokens?)\b|\b(?:failed|error|cancelled)\b|\b(?:done|ready|created|added|downloaded)\b)/gi;

function highlight(message: string): ReactNode {
  const access = ACCESS.exec(message);
  if (access) {
    const [, client, method, path, status] = access;
    const kind = status >= "500" ? "bad" : status >= "400" ? "warn" : "ok";
    return (
      <>
        <span className={`log-method ${method.toLowerCase()}`}>{method}</span>{" "}
        <span className="log-path">{path}</span>{" "}
        <span className={`log-status ${kind}`}>{status}</span>{" "}
        <span className="log-dim">{client}</span>
      </>
    );
  }
  return message.split(TOKENS).map((part, i) => {
    if (i % 2 === 0) return <Fragment key={i}>{part}</Fragment>;
    const lower = part.toLowerCase();
    const kind = /^https?:/.test(part)
      ? "log-url"
      : ["failed", "error", "cancelled"].includes(lower)
        ? "log-bad"
        : ["done", "ready", "created", "added", "downloaded"].includes(lower)
          ? "log-good"
          : "log-number";
    return (
      <span key={i} className={kind}>
        {part}
      </span>
    );
  });
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** The engine's log (also for a remote engine or a RunPod pod) and the app's own, newest last. */
export function Logs({
  client,
  engine,
}: {
  client: EngineClient | null;
  engine: EngineState;
}) {
  const [source, setSource] = useState<Source>("engine");
  const [minLevel, setMinLevel] = useState<Level>("INFO");
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [file, setFile] = useState("");
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    try {
      const result =
        source === "app"
          ? await window.tinylm.appLog(SHOWN)
          : client
            ? await client.logs(SHOWN)
            : null;
      if (!result) {
        setEntries([]);
        setFile("");
        setError("The engine is not connected: its log shows once it is.");
        return;
      }
      setEntries(result.lines.map(parse));
      setFile(result.file);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [source, client]);

  useEffect(() => {
    load();
    if (!follow) return;
    const timer = setInterval(load, 2000);
    return () => clearInterval(timer);
  }, [load, follow]);

  const needle = query.trim().toLowerCase();
  const shown = entries.filter(
    (e) =>
      RANK[e.level] >= RANK[minLevel] &&
      (!needle ||
        `${e.scope} ${e.message} ${e.more}`.toLowerCase().includes(needle)),
  );
  const counts = {
    warnings: entries.filter((e) => e.level === "WARNING").length,
    errors: entries.filter((e) => RANK[e.level] >= RANK.ERROR).length,
  };

  useEffect(() => {
    if (follow) bottom.current?.scrollIntoView({ block: "end" });
  }, [shown.length, follow, source]);

  const where = engine.phase === "ready" ? engine.where : null;
  return (
    <div className="settings-form logs">
      <h3>Logs</h3>
      <p className="muted">
        What the engine and this app did, newest last. The engine's log is read
        from the engine itself, so it is the pod's when the engine runs on
        RunPod. Successful reads (the app checks status every few seconds) are
        left out; set <code>TINYLM_LOG_LEVEL=DEBUG</code> to see them.
      </p>
      <div className="logs-toolbar">
        <div className="segmented">
          <button
            className={source === "engine" ? "active" : ""}
            onClick={() => setSource("engine")}
          >
            <Icon name="server" />
            Engine{where ? ` · ${where}` : ""}
          </button>
          <button
            className={source === "app" ? "active" : ""}
            onClick={() => setSource("app")}
          >
            <Icon name="device-desktop" />
            App
          </button>
        </div>
        <select
          value={minLevel}
          onChange={(e) => setMinLevel(e.target.value as Level)}
          title="The least important entries shown"
        >
          <option value="DEBUG">Everything</option>
          <option value="INFO">Info and above</option>
          <option value="WARNING">Warnings and errors</option>
          <option value="ERROR">Errors only</option>
        </select>
        <input
          placeholder="Filter, e.g. crawl or games"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label
          className="check"
          title="Reload every 2 seconds and keep the newest entry in view"
        >
          <input
            type="checkbox"
            checked={follow}
            onChange={(e) => setFollow(e.target.checked)}
          />{" "}
          Follow
        </label>
        <button onClick={load}><Icon name="refresh" />Refresh</button>
        {source === "app" && (
          <button onClick={() => window.tinylm.openLogFolder()}>
            <Icon name="folder-open" />Open folder
          </button>
        )}
      </div>
      <div className="logs-summary muted">
        <span>{plural(entries.length, "entry", "entries")}</span>
        {counts.warnings > 0 && (
          <span className="log-level warning">
            <Icon name="alert-triangle" />
            {plural(counts.warnings, "warning")}
          </span>
        )}
        {counts.errors > 0 && (
          <span className="log-level error">
            <Icon name="alert-circle" />
            {plural(counts.errors, "error")}
          </span>
        )}
        {file && <code title="The log file">{file}</code>}
      </div>
      {error && <p className="error">{error}</p>}
      <div className="log-view">
        {shown.length === 0 && !error && (
          <p className="muted">
            Nothing to show{needle ? " for this filter" : ""}.
          </p>
        )}
        {shown.map((e, i) => (
          <div key={i} className={`log-entry ${e.level.toLowerCase()}`}>
            <span className="log-time" title={`${e.date} ${e.time}`}>
              {e.time}
            </span>
            <span className={`log-level ${e.level.toLowerCase()}`}>
              <Icon name={LEVEL_ICONS[e.level]} />
              {e.level === "WARNING" ? "WARN" : e.level}
            </span>
            <span className={`log-scope ${scopeColor(e.scope)}`}>
              {e.scope}
            </span>
            <span className="log-message">
              {highlight(e.message)}
              {e.more && <pre className="log-more">{e.more}</pre>}
            </span>
          </div>
        ))}
        <div ref={bottom} />
      </div>
    </div>
  );
}
