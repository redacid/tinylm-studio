import { FormEvent, useCallback, useEffect, useState } from "react";
import { ChatMessage, Dataset, EngineClient, Example, ExampleFilter } from "./api";
import { GeneratePanel } from "./Generate";
import { useTabActive } from "./tabActive";
import { Icon } from "./Icon";
import { BusySpinner, useBusy } from "./Busy";

const PAGE_SIZE = 20;
const NEW_EXAMPLE: ChatMessage[] = [
  { role: "user", content: "" },
  { role: "assistant", content: "" },
];

/** Training datasets: import examples from files, browse, edit and split them. */
export function Datasets({ client }: { client: EngineClient }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState("");
  const [newName, setNewName] = useState("");
  const [error, setError] = useState<string | null>(null);
  // Bumped after anything that changes examples, so the list reloads.
  const [version, setVersion] = useState(0);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  const loading = useBusy();
  const { run: load } = loading;

  const refresh = useCallback(async () => {
    setVersion((v) => v + 1);
    const list = await load("Loading datasets…", () => client.datasets());
    setDatasets(list);
    setSelected((current) => (list.some((d) => d.name === current) ? current : list[0]?.name ?? ""));
  }, [client, load]);

  const active = useTabActive();
  useEffect(() => {
    if (active) refresh().catch(report);
  }, [refresh, active]);

  async function create(event: FormEvent) {
    event.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setError(null);
    try {
      await client.createDataset(name, 10);
      setNewName("");
      setSelected(name);
      await refresh();
    } catch (e) {
      report(e);
    }
  }

  const dataset = datasets.find((d) => d.name === selected);
  return (
    <div className="data">
      <aside>
        <strong>Datasets</strong>
        <BusySpinner busy={loading} />
        {datasets.map((d) => (
          <button key={d.name} className={d.name === selected ? "active" : ""} onClick={() => setSelected(d.name)}>
            {d.name}
            <div className="muted">
              {d.examples} examples · {d.train} train · {d.eval} eval
            </div>
          </button>
        ))}
        <form onSubmit={create}>
          <input placeholder="New dataset" value={newName} onChange={(e) => setNewName(e.target.value)} />
          <button type="submit" disabled={!newName.trim()}><Icon name="plus" />Add</button>
        </form>
      </aside>
      <section>
        {error && <p className="error">{error}</p>}
        {!dataset ? (
          !loading.busy && <p className="muted">
            A dataset holds examples of conversations: questions and the answers the model should learn to give.
            Create one, then import a JSONL or CSV file or add examples by hand.
          </p>
        ) : (
          <DatasetView
            key={dataset.name}
            client={client}
            dataset={dataset}
            version={version}
            onChange={() => refresh().catch(report)}
          />
        )}
      </section>
    </div>
  );
}

function DatasetView({
  client,
  dataset,
  version,
  onChange,
}: {
  client: EngineClient;
  dataset: Dataset;
  version: number;
  onChange: () => void;
}) {
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [split, setSplit] = useState<ExampleFilter["split"]>("all");
  const [origin, setOrigin] = useState<ExampleFilter["origin"]>(undefined);
  const [generating, setGenerating] = useState(false);
  const [offset, setOffset] = useState(0);
  const [examples, setExamples] = useState<Example[]>([]);
  const [total, setTotal] = useState(0);
  const [evalPercent, setEvalPercent] = useState(dataset.eval_percent);
  // The example being edited: its id, "new", or null.
  const [editing, setEditing] = useState<number | "new" | null>(null);
  const [draft, setDraft] = useState<ChatMessage[]>(NEW_EXAMPLE);
  // Loading the examples, and the long operations: importing, exporting, deleting.
  const loading = useBusy();
  const { run: load } = loading;
  const busy = useBusy();
  const [log, setLog] = useState<string[]>([]);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(query.trim());
      setOffset(0);
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    let stale = false;
    load("Loading examples…", () => client.examples(dataset.name, { query: search, split, origin, offset, limit: PAGE_SIZE }))
      .then((page) => {
        if (stale) return;
        setExamples(page.data);
        setTotal(page.total);
        if (page.data.length === 0 && offset > 0) setOffset(Math.max(0, offset - PAGE_SIZE));
      })
      .catch((e) => !stale && report(e));
    return () => {
      stale = true;
    };
  }, [client, dataset.name, search, split, origin, offset, version, load]);

  async function importFiles(files: FileList | null) {
    if (!files) return;
    setError(null);
    const lines: string[] = [];
    const list = Array.from(files);
    await busy.run("Importing…", async (progress) => {
      for (const [i, file] of list.entries()) {
        progress(`Importing ${i + 1} of ${list.length}: ${file.name}`);
        try {
          const r = await client.importExamples(dataset.name, file);
          if (r.error) {
            lines.push(`✗ ${file.name}: ${r.error}`);
          } else {
            lines.push(`✓ ${file.name}: ${r.added} added, ${r.duplicates} already there, ${r.invalid} invalid`);
            for (const e of r.errors ?? []) lines.push(`    line ${e.line}: ${e.error}`);
            if ((r.invalid ?? 0) > (r.errors?.length ?? 0)) lines.push(`    … and ${(r.invalid ?? 0) - (r.errors?.length ?? 0)} more`);
          }
        } catch (e) {
          lines.push(`✗ ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
        }
        setLog([...lines]);
      }
    });
    onChange();
  }

  // Saves the examples to a file, to import on another engine (a RunPod pod, say).
  async function exportDataset() {
    setError(null);
    try {
      const blob = await busy.run(`Exporting ${dataset.name}…`, () => client.exportDataset(dataset.name));
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${dataset.name}.jsonl`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
    } catch (e) {
      report(e);
    }
  }

  async function saveSplit() {
    if (evalPercent === dataset.eval_percent) return;
    await client.updateDataset(dataset.name, evalPercent).catch(report);
    onChange();
  }

  function edit(example: Example | null) {
    setError(null);
    setEditing(example ? example.id : "new");
    setDraft(example ? example.messages.map((m) => ({ ...m })) : NEW_EXAMPLE.map((m) => ({ ...m })));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      if (editing === "new") {
        const r = await client.addExample(dataset.name, draft);
        if (r.duplicate) throw new Error("The same example is already in the dataset.");
      } else if (editing !== null) {
        await client.editExample(dataset.name, editing, draft);
      }
      setEditing(null);
      onChange();
    } catch (e) {
      report(e);
    }
  }

  async function remove(id: number) {
    await client.deleteExample(dataset.name, id).catch(report);
    onChange();
  }

  async function removeDataset() {
    setConfirmDelete(false);
    await busy.run(`Deleting ${dataset.name}…`, () => client.deleteDataset(dataset.name)).catch(report);
    onChange();
  }

  const setMessage = (i: number, patch: Partial<ChatMessage>) =>
    setDraft((d) => d.map((m, j) => (j === i ? { ...m, ...patch } : m)));
  const last = Math.min(offset + PAGE_SIZE, total);

  return (
    <>
      <div className="actions">
        <label>
          <span className="button"><Icon name="file-import" />Import file</span>
          <input type="file" multiple accept=".jsonl,.json,.csv" onChange={(e) => importFiles(e.target.files)} disabled={busy.busy} />
        </label>
        <button
          onClick={exportDataset}
          disabled={busy.busy || dataset.examples === 0}
          title="Saves the examples to a JSONL file, for Import file on another engine, e.g. a RunPod pod"
        >
          <Icon name="file-export" />Export
        </button>
        <button onClick={() => edit(null)} disabled={editing !== null}><Icon name="plus" />Add example</button>
        <button onClick={() => setGenerating(!generating)}><Icon name="sparkles" />Generate</button>
        <label title="Examples held out from training to measure the model on data it has not seen">
          Evaluation{" "}
          <input
            type="number"
            className="narrow"
            min={0}
            max={50}
            value={evalPercent}
            onChange={(e) => setEvalPercent(+e.target.value)}
            onBlur={saveSplit}
          />
          %
        </label>
        <span className="muted">{dataset.train} for training · {dataset.eval} held out</span>
        <BusySpinner busy={busy} />
        {confirmDelete ? (
          <span className="danger">
            Delete "{dataset.name}" and all its examples?{" "}
            <button onClick={removeDataset}><Icon name="trash" />Delete</button> <button onClick={() => setConfirmDelete(false)}><Icon name="x" />Cancel</button>
          </span>
        ) : (
          <button className="danger" onClick={() => setConfirmDelete(true)} disabled={busy.busy}><Icon name="trash" />Delete dataset</button>
        )}
      </div>
      <p className="muted">
        JSONL, JSON or CSV. Each record is a chat (<code>messages</code> with system, user and assistant roles) or a pair:{" "}
        <code>prompt</code>/<code>completion</code>, <code>question</code>/<code>answer</code> or{" "}
        <code>instruction</code>/<code>input</code>/<code>output</code>. The model learns to write the assistant's answers.
      </p>
      {generating && (
        <GeneratePanel client={client} dataset={dataset.name} onAdded={onChange} onClose={() => setGenerating(false)} />
      )}
      {log.length > 0 && <pre className="muted">{log.join("\n")}</pre>}
      {error && <p className="error">{error}</p>}
      {editing !== null && (
        <form className="example-editor" onSubmit={save}>
          {draft.map((m, i) => (
            <div key={i} className="message-edit">
              <select value={m.role} onChange={(e) => setMessage(i, { role: e.target.value as ChatMessage["role"] })}>
                <option value="system">system</option>
                <option value="user">user</option>
                <option value="assistant">assistant</option>
              </select>
              <textarea value={m.content} onChange={(e) => setMessage(i, { content: e.target.value })} />
              <button type="button" onClick={() => setDraft((d) => d.filter((_, j) => j !== i))} disabled={draft.length <= 2} title="Remove this message">
                <Icon name="x" />
              </button>
            </div>
          ))}
          <div className="buttons">
            <button
              type="button"
              onClick={() =>
                setDraft((d) => [...d, { role: d[d.length - 1]?.role === "user" ? "assistant" : "user", content: "" }])
              }
            >
              <Icon name="plus" />
              Add message
            </button>
            <button type="button" onClick={() => setEditing(null)}><Icon name="x" />Cancel</button>
            <button type="submit" disabled={draft.some((m) => !m.content.trim())}><Icon name="device-floppy" />Save</button>
          </div>
        </form>
      )}
      <div className="documents">
        <div className="filters">
          <input type="search" placeholder="Search examples" value={query} onChange={(e) => setQuery(e.target.value)} />
          <select
            value={split}
            onChange={(e) => {
              setSplit(e.target.value as ExampleFilter["split"]);
              setOffset(0);
            }}
          >
            <option value="all">All examples</option>
            <option value="train">Training</option>
            <option value="eval">Held out</option>
          </select>
          <select
            value={origin ?? ""}
            onChange={(e) => {
              setOrigin((e.target.value || undefined) as ExampleFilter["origin"]);
              setOffset(0);
            }}
          >
            <option value="">Any origin</option>
            <option value="manual">Written by hand</option>
            <option value="imported">Imported</option>
            <option value="generated">Generated</option>
          </select>
          <span className="pager">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}><Icon name="chevron-left" />Previous</button>
            <span className="muted">{total === 0 ? "0" : `${offset + 1}–${last}`} of {total}</span>
            <button disabled={last >= total} onClick={() => setOffset(offset + PAGE_SIZE)}>Next<Icon name="chevron-right" /></button>
          </span>
        </div>
        {examples.map((example) => (
          <div key={example.id} className="example">
            <div className="example-head">
              <span className={`badge ${example.split}`}>{example.split === "eval" ? "held out" : "training"}</span>
              {example.origin !== "manual" && <span className="badge">{example.origin}</span>}
              <span className="muted">#{example.id}</span>
              {example.source && (
                <span className="muted source" title={example.source}>
                  from {example.source}
                </span>
              )}
              <span className="buttons">
                <button onClick={() => edit(example)} disabled={editing !== null}><Icon name="pencil" />Edit</button>
                <button onClick={() => remove(example.id)}><Icon name="trash" />Delete</button>
              </span>
            </div>
            {example.messages.map((m, i) => (
              <div key={i} className={`turn ${m.role}`}>
                <span className="role">{m.role}</span>
                <span className="text">{m.content.length > 600 ? `${m.content.slice(0, 600)}…` : m.content}</span>
              </div>
            ))}
          </div>
        ))}
        {total === 0 && !loading.busy && <p className="muted">{search || split !== "all" || origin ? "Nothing matches." : "No examples yet."}</p>}
      </div>
    </>
  );
}
