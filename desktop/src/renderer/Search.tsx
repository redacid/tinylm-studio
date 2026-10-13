import { FormEvent, useEffect, useState } from "react";
import { CrawlJob, EngineClient, SearchKind, SearchOutcome, SearchResult, SearchSettings } from "./api";
import { PROVIDER_TITLES } from "./SearchSettings";
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
  const [outcome, setOutcome] = useState<SearchOutcome | null>(null);
  const [searchSettings, setSearchSettings] = useState<SearchSettings | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  // Video links already added, marked in the list so it is clear what happened.
  const [linked, setLinked] = useState<Map<string, Mark>>(new Map());
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<CrawlJob | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  useEffect(() => {
    client.searchSettings().then(setSearchSettings, () => {});
  }, [client]);

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
      const outcome = await client.search(query.trim(), kind, language);
      const found = outcome.data;
      setOutcome(outcome);
      setResults(found);
      setSelected(new Set(found.map((r) => r.url)));
      setLinked(new Map());
    } catch (e) {
      setResults(null);
      report(e);
    } finally {
      setBusy(false);
    }
  }

  const chosen = (results ?? []).filter((r) => selected.has(r.url));

  // When the page crawl ends, untick the pages that are now in the collection.
  const finished = job !== null && job.state !== "running";
  useEffect(() => {
    if (!finished || !job) return;
    const done = new Set(Object.entries(job.outcomes).filter(([, o]) => o !== "failed").map(([url]) => url));
    setSelected((current) => new Set([...current].filter((url) => !done.has(stripFragment(url)))));
  }, [finished, job]);

  /** How a page result fared in the crawl started from these results, if any. */
  function pageMark(url: string): Mark | null {
    if (!job || !job.options.urls.includes(url)) return null;
    const outcome = job.outcomes[stripFragment(url)];
    if (outcome) return MARKS[outcome];
    if (job.state !== "running") return MARKS.unreached;
    return job.current === url ? MARKS.downloading : MARKS.waiting;
  }

  async function addPages() {
    setError(null);
    setMessage(null);
    try {
      const urls = chosen.map((r) => r.url);
      // Only these pages: depth 0, and a page budget that covers all of them.
      setJob(await client.startCrawl(collection, { urls, max_pages: urls.length, max_depth: 0, use_browser: false }));
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
        await client.addWebsite(collection, { name: site, urls, use_browser: false });
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
      setLinked((current) => {
        const next = new Map(current);
        results.forEach((r, i) => next.set(chosen[i].url, r.duplicate ? MARKS.duplicate : MARKS.added));
        return next;
      });
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
      <SearchSummary settings={searchSettings} outcome={outcome} />
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
                    <MarkLabel mark={kind === "video" ? linked.get(r.url) ?? null : pageMark(r.url)} />
                    <div className="muted">{r.url}</div>
                    {r.snippet && <div className="snippet">{r.snippet}</div>}
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <div className="buttons">
            {(message ?? pagesNotice(job, collection)) && <span className="notice">{message ?? pagesNotice(job, collection)}</span>}
            {kind === "web" ? (
              <>
                <button onClick={saveSites} disabled={chosen.length === 0} title="Save the sites of the selected pages, to crawl them further now and later">
                  Save as websites
                </button>
                <button onClick={addPages} disabled={chosen.length === 0 || job?.state === "running"}>
                  {job?.state === "running" ? "Adding…" : `Add ${chosen.length} page${chosen.length === 1 ? "" : "s"}`}
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

/** Which provider searches and which sites are filtered, so a short list is not a mystery. */
function SearchSummary({ settings, outcome }: { settings: SearchSettings | null; outcome: SearchOutcome | null }) {
  const source = outcome ?? settings;
  if (!source) return null;
  const parts = [`Searching with ${PROVIDER_TITLES[source.provider]}`];
  if (source.include_domains.length) parts.push(`only ${source.include_domains.join(", ")}`);
  if (source.exclude_domains.length) parts.push(`not ${source.exclude_domains.join(", ")}`);
  if (outcome?.hidden) parts.push(`${outcome.hidden} result${outcome.hidden === 1 ? "" : "s"} hidden by the filter`);
  return <p className="muted search-summary">{parts.join(" · ")} (Settings › Search)</p>;
}

type Mark = { text: string; tone: "ok" | "bad" | "muted" };

const MARKS = {
  added: { text: "✓ added", tone: "ok" },
  duplicate: { text: "✓ already there", tone: "ok" },
  known: { text: "✓ already there", tone: "ok" },
  skipped: { text: "skipped: too little text, or not allowed by robots.txt", tone: "muted" },
  failed: { text: "✗ could not download", tone: "bad" },
  downloading: { text: "downloading…", tone: "muted" },
  waiting: { text: "waiting", tone: "muted" },
  unreached: { text: "not downloaded", tone: "muted" },
} satisfies Record<string, Mark>;

function MarkLabel({ mark }: { mark: Mark | null }) {
  return mark ? <span className={`mark ${mark.tone}`}> {mark.text}</span> : null;
}

function stripFragment(url: string): string {
  return url.split("#")[0];
}

/** One line about the page crawl started from the search results. */
function pagesNotice(job: CrawlJob | null, collection: string): string | null {
  if (!job || job.website_id !== null) return null;
  const total = job.options.urls.length;
  if (job.state === "running") return `Downloading ${Math.min(Object.keys(job.outcomes).length + 1, total)} of ${total}…`;
  const count = (kinds: string[]) => Object.values(job.outcomes).filter((o) => kinds.includes(o)).length;
  const parts = [`Added ${count(["added"])} page${count(["added"]) === 1 ? "" : "s"} to "${collection}"`];
  if (count(["duplicate", "known"])) parts.push(`${count(["duplicate", "known"])} already there`);
  if (count(["skipped"])) parts.push(`${count(["skipped"])} skipped`);
  if (count(["failed"])) parts.push(`${count(["failed"])} failed`);
  if (job.state === "cancelled") parts.push("stopped");
  return parts.join("; ") + ".";
}
