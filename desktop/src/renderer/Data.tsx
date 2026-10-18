import { FormEvent, useCallback, useEffect, useState } from "react";
import { Collection, EngineClient } from "./api";
import { Documents } from "./Documents";
import { Search } from "./Search";
import { Websites } from "./Websites";
import { useTabActive } from "./tabActive";
import { Icon } from "./Icon";
import { BusySpinner, useBusy } from "./Busy";

export function Data({ client }: { client: EngineClient }) {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState("");
  const [newName, setNewName] = useState("");
  // Loading the list, and the long operations: adding files, importing, exporting.
  const loading = useBusy();
  const busy = useBusy();
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  // In-app confirmation: a native confirm() dialog leaves text inputs unfocusable in Electron on Linux.
  const [confirmDelete, setConfirmDelete] = useState(false);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  // Bumped after anything that adds or removes documents, so the document list reloads.
  const [version, setVersion] = useState(0);
  // Bumped when websites are saved from search results, so the website list reloads.
  const [sitesVersion, setSitesVersion] = useState(0);

  const { run: load } = loading;
  const refresh = useCallback(async () => {
    setVersion((v) => v + 1);
    const list = await load("Loading collections…", () => client.collections());
    setCollections(list);
    setSelected((current) => (list.some((c) => c.name === current) ? current : list[0]?.name ?? ""));
  }, [client, load]);

  const active = useTabActive();
  useEffect(() => {
    if (active) refresh().catch(report);
  }, [refresh, active]);

  const refreshAll = useCallback(() => {
    refresh().catch(report);
  }, [refresh]);

  async function create(event: FormEvent) {
    event.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setError(null);
    try {
      await client.createCollection(name);
      setNewName("");
      setSelected(name);
      await refresh();
    } catch (e) {
      report(e);
    }
  }

  async function upload(files: FileList | null) {
    if (!files || !selected) return;
    setError(null);
    const lines: string[] = [];
    // One file per request, so progress is visible and a bad file does not block the rest.
    const list = Array.from(files);
    await busy.run("Indexing…", async (progress) => {
      for (const [i, file] of list.entries()) {
        progress(`Indexing ${i + 1} of ${list.length}: ${file.name}`);
        try {
          const r = await client.upload(selected, file);
          lines.push(r.error ? `✗ ${file.name}: ${r.error}` : r.duplicate ? `= ${file.name}: already added` : `✓ ${file.name}: ${r.chunks} chunks`);
        } catch (e) {
          lines.push(`✗ ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
        }
        setLog([...lines]);
      }
    });
    await refresh().catch(report);
  }

  // Saves the collection to a file, to import on another engine (a RunPod pod, say).
  async function exportCollection() {
    setError(null);
    try {
      const blob = await busy.run(`Exporting ${selected}…`, () => client.exportCollection(selected));
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${selected}.tinylm.jsonl`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
    } catch (e) {
      report(e);
    }
  }

  // The engine embeds the imported chunks in the background; this follows it until it is done.
  async function importCollection(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setError(null);
    try {
      const job = await busy.run(`Importing ${file.name}…`, async (progress) => {
        let job = await client.importCollection(file);
        while (job.state === "running") {
          const done = job.added + job.duplicates;
          progress(`Importing ${job.collection}: ${done}${job.total !== null ? ` of ${job.total}` : ""} documents`);
          await new Promise((resolve) => setTimeout(resolve, 1000));
          job = await client.importJob(job.id);
        }
        return job;
      });
      if (job.state === "error") throw new Error(`Import into ${job.collection} failed: ${job.error}`);
      setLog([`✓ ${file.name} → ${job.collection}: ${job.added} documents added, ${job.duplicates} already there`]);
      setSelected(job.collection);
    } catch (e) {
      report(e);
    } finally {
      await refresh().catch(report);
    }
  }

  async function removeCollection() {
    if (!selected) return;
    setConfirmDelete(false);
    await busy.run(`Deleting ${selected}…`, () => client.deleteCollection(selected)).catch(report);
    await refresh().catch(report);
  }

  async function removeDocument(id: number) {
    await client.deleteDocument(selected, id).catch(report);
    await refresh().catch(report);
  }

  return (
    <div className="data">
      <aside>
        <strong>Collections</strong>
        <BusySpinner busy={loading} />
        {collections.map((c) => (
          <button key={c.name} className={c.name === selected ? "active" : ""} onClick={() => { setSelected(c.name); setConfirmDelete(false); }}>
            {c.name}
            <div className="muted">{c.documents} documents · {c.chunks} chunks</div>
          </button>
        ))}
        <form onSubmit={create}>
          <input placeholder="New collection" value={newName} onChange={(e) => setNewName(e.target.value)} />
          <button type="submit" disabled={!newName.trim()}><Icon name="plus" />Add</button>
        </form>
        <label title="A file saved with Export on another engine: this computer's, a server's or a RunPod pod's">
          <span className="button"><Icon name="file-import" />Import collection</span>
          <input type="file" accept=".jsonl" onChange={(e) => { importCollection(e.target.files); e.target.value = ""; }} disabled={busy.busy} />
        </label>
      </aside>
      <section>
        {error && <p className="error">{error}</p>}
        {!selected && <BusySpinner busy={busy} block />}
        {!selected ? (
          collections.length === 0 && !loading.busy && <p className="muted">Create a collection, then add files to it, or import one exported from another engine. Chat can answer from a collection and cite it.</p>
        ) : (
          <>
            <div className="actions">
              <label>
                <span className="button"><Icon name="file-plus" />Add files</span>
                <input type="file" multiple onChange={(e) => upload(e.target.files)} disabled={busy.busy} />
              </label>
              <label>
                <span className="button"><Icon name="folder-open" />Add folder</span>
                {/* @ts-expect-error webkitdirectory is supported by Chromium (Electron) */}
                <input type="file" webkitdirectory="" onChange={(e) => upload(e.target.files)} disabled={busy.busy} />
              </label>
              <button onClick={exportCollection} disabled={busy.busy} title="Saves its documents to a file, for Import collection on another engine, e.g. a RunPod pod">
                <Icon name="file-export" />Export
              </button>
              <BusySpinner busy={busy} />
              {confirmDelete ? (
                <span className="danger">
                  Delete "{selected}" and all its documents?{" "}
                  <button onClick={removeCollection}><Icon name="trash" />Delete</button>{" "}
                  <button onClick={() => setConfirmDelete(false)}><Icon name="x" />Cancel</button>
                </span>
              ) : (
                <button className="danger" onClick={() => setConfirmDelete(true)} disabled={busy.busy}>
                  <Icon name="trash" />Delete collection
                </button>
              )}
            </div>
            <p className="muted">PDF, DOCX, PPTX, XLSX, CSV, HTML, Markdown, text and JSON files.</p>
            {log.length > 0 && <pre className="muted">{log.join("\n")}</pre>}
            <Search
              key={`search-${selected}`}
              client={client}
              collection={selected}
              onSaved={() => {
                setSitesVersion((v) => v + 1);
                refreshAll();
              }}
            />
            <Websites client={client} collection={selected} onIndexed={refreshAll} savedVersion={sitesVersion} />
            <Documents key={selected} client={client} collection={selected} version={version} onDelete={removeDocument} />
          </>
        )}
      </section>
    </div>
  );
}
