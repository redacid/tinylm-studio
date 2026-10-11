import { FormEvent, useCallback, useEffect, useState } from "react";
import { Collection, DocumentInfo, EngineClient } from "./api";

export function Data({ client }: { client: EngineClient }) {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState("");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [newName, setNewName] = useState("");
  const [progress, setProgress] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  const refresh = useCallback(async () => {
    const list = await client.collections();
    setCollections(list);
    setSelected((current) => (list.some((c) => c.name === current) ? current : list[0]?.name ?? ""));
  }, [client]);

  useEffect(() => {
    refresh().catch(report);
  }, [refresh]);

  useEffect(() => {
    if (selected) client.documents(selected).then(setDocuments, report);
    else setDocuments([]);
  }, [client, selected, collections]);

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
    for (const [i, file] of list.entries()) {
      setProgress(`Indexing ${i + 1} of ${list.length}: ${file.name}`);
      try {
        const r = await client.upload(selected, file);
        lines.push(r.error ? `✗ ${file.name}: ${r.error}` : r.duplicate ? `= ${file.name}: already added` : `✓ ${file.name}: ${r.chunks} chunks`);
      } catch (e) {
        lines.push(`✗ ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
      }
      setLog([...lines]);
    }
    setProgress(null);
    await refresh().catch(report);
  }

  async function removeCollection() {
    if (!selected || !confirm(`Delete the collection "${selected}" and all its documents?`)) return;
    await client.deleteCollection(selected).catch(report);
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
        {collections.map((c) => (
          <button key={c.name} className={c.name === selected ? "active" : ""} onClick={() => setSelected(c.name)}>
            {c.name}
            <div className="muted">{c.documents} documents · {c.chunks} chunks</div>
          </button>
        ))}
        <form onSubmit={create}>
          <input placeholder="New collection" value={newName} onChange={(e) => setNewName(e.target.value)} />
          <button type="submit" disabled={!newName.trim()}>Add</button>
        </form>
      </aside>
      <section>
        {error && <p className="error">{error}</p>}
        {!selected ? (
          <p className="muted">Create a collection, then add files to it. Chat can answer from a collection and cite it.</p>
        ) : (
          <>
            <div className="actions">
              <label>
                <span className="button">Add files</span>
                <input type="file" multiple onChange={(e) => upload(e.target.files)} disabled={!!progress} />
              </label>
              <label>
                <span className="button">Add folder</span>
                {/* @ts-expect-error webkitdirectory is supported by Chromium (Electron) */}
                <input type="file" webkitdirectory="" onChange={(e) => upload(e.target.files)} disabled={!!progress} />
              </label>
              {progress && <span className="muted">{progress}</span>}
              <button className="danger" onClick={removeCollection} disabled={!!progress}>Delete collection</button>
            </div>
            <p className="muted">PDF, DOCX, PPTX, XLSX, CSV, HTML, Markdown, text and JSON files.</p>
            {log.length > 0 && <pre className="muted">{log.join("\n")}</pre>}
            <table>
              <thead>
                <tr><th>Document</th><th>Size</th><th>Chunks</th><th /></tr>
              </thead>
              <tbody>
                {documents.map((d) => (
                  <tr key={d.id}>
                    <td>{d.name}</td>
                    <td>{Math.ceil(d.chars / 1000)}k chars</td>
                    <td>{d.chunks}</td>
                    <td><button onClick={() => removeDocument(d.id)}>Delete</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </div>
  );
}
