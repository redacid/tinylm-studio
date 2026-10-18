import { KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import { DataSchema, EngineClient, QueryResult } from "./api";
import { BusySpinner, useBusy } from "./Busy";
import { Icon } from "./Icon";
import { useTabActive } from "./tabActive";

const QUERY_KEY = "tinylm.sqlQuery";
const LIMITS = [100, 500, 1000, 10000];

function savedQuery(): string {
  try {
    return localStorage.getItem(QUERY_KEY) ?? "";
  } catch {
    return "";
  }
}

/** A cell as text: NULL stands out, numbers and text as they are. */
function Cell({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="muted">NULL</span>;
  const text = String(value);
  return <span title={text.length > 200 ? text : undefined}>{text.length > 500 ? `${text.slice(0, 500)}…` : text}</span>;
}

/** The SQL tab: read-only queries over the data (collections, documents, chunks, websites). */
export function Query({ client }: { client: EngineClient }) {
  const [schema, setSchema] = useState<DataSchema | null>(null);
  const [sql, setSql] = useState(savedQuery);
  const [limit, setLimit] = useState(500);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loading = useBusy();
  const running = useBusy();
  const editor = useRef<HTMLTextAreaElement>(null);
  const { run: load } = loading;

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  // Read again when the tab opens: a website saved meanwhile adds its tables.
  const active = useTabActive();
  const refresh = useCallback(async () => {
    try {
      setSchema(await load("Loading the tables…", () => client.dataSchema()));
    } catch (e) {
      report(e);
    }
  }, [client, load]);
  useEffect(() => {
    if (active) refresh();
  }, [active, refresh]);

  useEffect(() => {
    try {
      localStorage.setItem(QUERY_KEY, sql);
    } catch {
      // Not kept: the query is only lost on a restart.
    }
  }, [sql]);

  async function run() {
    if (!sql.trim() || running.busy) return;
    setError(null);
    try {
      setResult(await running.run("Running the query…", () => client.query(sql, limit)));
    } catch (e) {
      setResult(null);
      report(e);
    }
  }

  function keys(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      run();
    }
  }

  /** Puts text where the cursor is in the query. */
  function insert(text: string) {
    const area = editor.current;
    if (!area) return setSql((q) => q + text);
    const { selectionStart: start, selectionEnd: end } = area;
    setSql(sql.slice(0, start) + text + sql.slice(end));
    requestAnimationFrame(() => {
      area.focus();
      area.setSelectionRange(start + text.length, start + text.length);
    });
  }

  return (
    <div className="data sql">
      <aside>
        <h3>Tables</h3>
        <p className="muted">Click a table or a field to put it in the query.</p>
        <BusySpinner busy={loading} />
        {schema?.tables.map((table) => (
          <details key={table.name} className="sql-table">
            <summary>
              <Icon name="table" />
              <button className="link" onClick={() => insert(table.name)} title="Put in the query">
                {table.name}
              </button>
            </summary>
            <p className="muted">{table.about}</p>
            <ul>
              {table.columns.map((column) => (
                <li key={column.name} title={column.about || undefined}>
                  <button className="link" onClick={() => insert(column.name)}>
                    {column.name}
                  </button>
                  <code>{column.type}</code>
                  {column.primary_key && <span className="badge">key</span>}
                  {column.not_null && !column.primary_key && <span className="badge">not null</span>}
                  {column.about && <div className="muted">{column.about}</div>}
                </li>
              ))}
            </ul>
          </details>
        ))}
        <h3>Examples</h3>
        {schema?.examples.map((example) => (
          <button key={example.title} onClick={() => setSql(example.sql)} title={example.sql}>
            <Icon name="braces" />
            {example.title}
          </button>
        ))}
      </aside>
      <section>
        <p className="muted">
          Only reading: a query is one SELECT over the tables on the left. Writing, other tables and PRAGMA are refused,
          and a query is stopped after 15 seconds.
        </p>
        <textarea
          ref={editor}
          className="sql-editor"
          spellCheck={false}
          placeholder="SELECT collection, count(*) FROM documents GROUP BY collection"
          value={sql}
          onChange={(e) => setSql(e.target.value)}
          onKeyDown={keys}
        />
        <div className="actions">
          <button className="primary" onClick={run} disabled={!sql.trim() || running.busy}>
            <Icon name="player-play" />
            Run
          </button>
          <span className="muted">Ctrl+Enter</span>
          <label>
            Rows at most{" "}
            <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
              {LIMITS.map((n) => (
                <option key={n} value={n}>
                  {n.toLocaleString()}
                </option>
              ))}
            </select>
          </label>
          <BusySpinner busy={running} />
          {result && !running.busy && (
            <span className="muted">
              {result.rows.length.toLocaleString()} {result.rows.length === 1 ? "row" : "rows"}
              {result.truncated && ` shown, there are more (raise the limit or add LIMIT)`} ·{" "}
              {result.elapsed_ms} ms
            </span>
          )}
        </div>
        {error && <p className="error">{error}</p>}
        {result && (
          <div className="sql-result">
            <table>
              <thead>
                <tr>
                  {result.columns.map((column, i) => (
                    <th key={i}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, i) => (
                  <tr key={i}>
                    {row.map((value, j) => (
                      <td key={j}>
                        <Cell value={value} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
