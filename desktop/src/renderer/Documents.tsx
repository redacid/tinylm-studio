import { useEffect, useState } from "react";
import { DocumentFilter, DocumentInfo, EngineClient } from "./api";
import { Icon } from "./Icon";
import { BusySpinner, useBusy } from "./Busy";

const PAGE_SIZE = 50;

/** A collection's documents, searchable and one page at a time. Key it by collection to reset. */
export function Documents({
  client,
  collection,
  version,
  onDelete,
}: {
  client: EngineClient;
  collection: string;
  /** Changes whenever documents were added or removed elsewhere, to reload the page. */
  version: number;
  onDelete: (id: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState<DocumentFilter["kind"]>("all");
  const [sort, setSort] = useState<DocumentFilter["sort"]>("newest");
  const [offset, setOffset] = useState(0);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const loading = useBusy();
  const { run: load } = loading;

  // Search as the user types, but not on every keystroke.
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(query.trim());
      setOffset(0);
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    let stale = false;
    load("Loading documents…", () => client.documents(collection, { query: search, kind, sort, offset, limit: PAGE_SIZE }))
      .then((page) => {
        if (stale) return;
        setDocuments(page.data);
        setTotal(page.total);
        setError(null);
        // The last page emptied by deletions: step back to the previous one.
        if (page.data.length === 0 && offset > 0) setOffset(Math.max(0, offset - PAGE_SIZE));
      })
      .catch((e) => !stale && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      stale = true;
    };
  }, [client, collection, search, kind, sort, offset, version, load]);

  const last = Math.min(offset + PAGE_SIZE, total);
  return (
    <div className="documents">
      <div className="filters">
        <input
          type="search"
          placeholder="Search by name or address"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={kind} onChange={(e) => {
            setKind(e.target.value as DocumentFilter["kind"]);
            setOffset(0);
          }}>
          <option value="all">All documents</option>
          <option value="file">Uploaded files</option>
          <option value="web">Web pages</option>
        </select>
        <select value={sort} onChange={(e) => {
            setSort(e.target.value as DocumentFilter["sort"]);
            setOffset(0);
          }}>
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="name">By name</option>
          <option value="largest">Largest first</option>
        </select>
        <Pager offset={offset} last={last} total={total} onChange={setOffset} />
      </div>
      {error && <p className="error">{error}</p>}
      <BusySpinner busy={loading} block />
      <table>
        <thead>
          <tr><th>Document</th><th>Added</th><th>Size</th><th>Chunks</th><th /></tr>
        </thead>
        <tbody>
          {documents.map((d) => (
            <tr key={d.id}>
              <td>
                {d.name}
                {d.source !== d.name && <div className="muted">{d.source}</div>}
              </td>
              <td className="nowrap">{new Date(d.created_at).toLocaleString()}</td>
              <td className="nowrap">{d.chars < 1000 ? `${d.chars} chars` : `${Math.round(d.chars / 1000)}k chars`}</td>
              <td>{d.chunks}</td>
              <td><button onClick={() => onDelete(d.id)}><Icon name="trash" />Delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      {total === 0 && !loading.busy && <p className="muted">{search || kind !== "all" ? "Nothing matches." : "No documents yet."}</p>}
      {total > PAGE_SIZE && (
        <div className="filters bottom">
          <Pager offset={offset} last={last} total={total} onChange={setOffset} />
        </div>
      )}
    </div>
  );
}

function Pager({
  offset,
  last,
  total,
  onChange,
}: {
  offset: number;
  last: number;
  total: number;
  onChange: (offset: number) => void;
}) {
  return (
    <span className="pager">
      <button disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - PAGE_SIZE))}><Icon name="chevron-left" />Previous</button>
      <span className="muted">{total === 0 ? "0" : `${offset + 1}–${last}`} of {total}</span>
      <button disabled={last >= total} onClick={() => onChange(offset + PAGE_SIZE)}>Next<Icon name="chevron-right" /></button>
    </span>
  );
}
