import { FormEvent, useEffect, useState } from "react";
import { CrawlJob, EngineClient, SearchKind, SearchResult } from "./api";
import { CrawlProgress } from "./Websites";

/**
 * Finds pages and videos on the web and adds them to a collection: pages are downloaded and
 * indexed (or saved as websites to crawl further), videos are kept as links answers can cite.
 */
export function Search({
  client,
  collection,
  onSaved,
}: {
  client: EngineClient;
  collection: string;
  onSaved: () => void;
}) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<SearchKind>("web");
  const [language, setLanguage] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  // Results already in the collection, marked in the list so it is clear what was added.
  const [added, setAdded] = useState<Set<string>>(new Set());
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<CrawlJob | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  // Follow the crawl started from here until it ends.
  useEffect(() => {
    if (job?.state !== "running") return;
    const timer = setInterval(async () => {
      const latest = (await client.crawls()).find((j) => j.id === job.id);
      if (latest) setJob(latest);
    }, 1500);
    return () => clearInterval(timer);
  }, [client, job?.id, job?.state]);

  async function search(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    if (job?.state !== "running") setJob(null);
    try {
      const found = await client.search(query.trim(), kind, language);
      setResults(found);
      setSelected(new Set(found.map((r) => r.url)));
      setAdded(new Set());
    } catch (e) {
      setResults(null);
      report(e);
    } finally {
      setBusy(false);
    }
  }

  const chosen = (results ?? []).filter((r) => selected.has(r.url));

  async function addPages() {
    setError(null);
    try {
      const urls = chosen.map((r) => r.url);
      // Only these pages: depth 0, and a page budget that covers all of them.
      setJob(await client.startCrawl(collection, { urls, max_pages: urls.length, max_depth: 0, per_minute: 20, use_browser: false }));
      setMessage(null);
    } catch (e) {
      report(e);
    }
  }

  async function saveSites() {
    setError(null);
    try {
      const bySite = new Map<string, string[]>();
      for (const r of chosen) bySite.set(r.site, [...(bySite.get(r.site) ?? []), r.url]);
      for (const [site, urls] of bySite) {
        await client.addWebsite(collection, { name: site, urls, max_pages: 50, max_depth: 2, per_minute: 20, use_browser: false });
      }
      onSaved();
      setMessage(`Saved ${bySite.size} website${bySite.size === 1 ? "" : "s"}: crawl ${bySite.size === 1 ? "it" : "them"} from the list below.`);
    } catch (e) {
      report(e);
    }
  }

  async function addLinks() {
    setError(null);
    setMessage(null);
    setAdding(true);
    try {
      const results = await client.addLinks(collection, chosen);
      const fresh = results.filter((r) => !r.duplicate).length;
      const known = results.length - fresh;
      setAdded((current) => new Set([...current, ...chosen.map((r) => r.url)]));
      setSelected(new Set());
      setMessage(
        `Added ${fresh} link${fresh === 1 ? "" : "s"} to "${collection}"` +
          (known ? `; ${known} ${known === 1 ? "was" : "were"} already there.` : "."),
      );
      onSaved();
    } catch (e) {
      report(e);
    } finally {
      setAdding(false);
    }
  }

  function toggle(url: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });
  }

  return (
    <div className="search">
      <form onSubmit={search}>
        <input
          type="search"
          placeholder={kind === "video" ? "Find videos, e.g. Elden Ring trailer" : "Find pages, e.g. Elden Ring review"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={kind} onChange={(e) => setKind(e.target.value as SearchKind)}>
          <option value="web">Pages</option>
          <option value="video">Videos</option>
        </select>
        <select value={language} onChange={(e) => setLanguage(e.target.value)} title="Language of the results">
          <option value="">Any language</option>
          <option value="uk">Українська</option>
          <option value="ru">Русский</option>
          <option value="en">English</option>
        </select>
        <button type="submit" disabled={busy || !query.trim()}>{busy ? "Searching…" : "Search"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      {results && results.length === 0 && <p className="muted">Nothing found.</p>}
      {results && results.length > 0 && (
        <>
          <ul className="results">
            {results.map((r) => (
              <li key={r.url}>
                <label>
                  <input type="checkbox" checked={selected.has(r.url)} onChange={() => toggle(r.url)} />
                  <span>
                    <strong>{r.title}</strong>
                    {added.has(r.url) && <span className="added"> ✓ added</span>}
                    <div className="muted">{r.url}</div>
                    {r.snippet && <div className="snippet">{r.snippet}</div>}
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <div className="buttons">
            {message && <span className="notice">{message}</span>}
            {kind === "web" ? (
              <>
                <button onClick={saveSites} disabled={chosen.length === 0} title="Save the sites of the selected pages, to crawl them further now and later">
                  Save as websites
                </button>
                <button onClick={addPages} disabled={chosen.length === 0 || job?.state === "running"}>
                  Add {chosen.length} page{chosen.length === 1 ? "" : "s"}
                </button>
              </>
            ) : (
              <button onClick={addLinks} disabled={chosen.length === 0 || adding} title="Keep title, address and description so answers can cite them">
                {adding ? "Adding…" : `Add ${chosen.length} link${chosen.length === 1 ? "" : "s"}`}
              </button>
            )}
          </div>
        </>
      )}
      {job && <CrawlProgress job={job} />}
    </div>
  );
}
