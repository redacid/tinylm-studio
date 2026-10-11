import { FormEvent, useEffect, useState } from "react";
import { CrawlJob, EngineClient } from "./api";

/** Adds pages of websites to a collection and shows the progress of crawls into it. */
export function Websites({
  client,
  collection,
  onIndexed,
}: {
  client: EngineClient;
  collection: string;
  onIndexed: () => void;
}) {
  const [urls, setUrls] = useState("");
  const [maxPages, setMaxPages] = useState(50);
  const [maxDepth, setMaxDepth] = useState(2);
  const [perMinute, setPerMinute] = useState(20);
  const [useBrowser, setUseBrowser] = useState(false);
  const [job, setJob] = useState<CrawlJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Show the latest crawl into this collection, and poll it while it runs.
  useEffect(() => {
    let indexed = -1;
    let stopped = false;
    async function poll() {
      const latest = (await client.crawls()).filter((j) => j.collection === collection).pop() ?? null;
      if (stopped) return;
      setJob(latest);
      if (latest && latest.indexed !== indexed) {
        if (indexed >= 0) onIndexed();
        indexed = latest.indexed;
      }
    }
    poll().catch(() => {});
    const timer = setInterval(() => poll().catch(() => {}), 1500);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [client, collection, onIndexed]);

  async function start(event: FormEvent) {
    event.preventDefault();
    const list = urls.split(/\s+/).filter(Boolean);
    if (list.length === 0) return;
    setError(null);
    try {
      setJob(
        await client.startCrawl(collection, {
          urls: list,
          max_pages: maxPages,
          max_depth: maxDepth,
          per_minute: perMinute,
          use_browser: useBrowser,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const running = job?.state === "running";
  return (
    <div className="websites">
      <form onSubmit={start}>
        <textarea
          placeholder={"Website addresses, one per line\nhttps://example.com/reviews/"}
          value={urls}
          onChange={(e) => setUrls(e.target.value)}
        />
        <div className="crawl-options">
          <label>
            Pages <input type="number" min={1} value={maxPages} onChange={(e) => setMaxPages(+e.target.value)} />
          </label>
          <label title="How many links away from the start pages to go">
            Depth <input type="number" min={0} max={10} value={maxDepth} onChange={(e) => setMaxDepth(+e.target.value)} />
          </label>
          <label title="Requests per minute to each site. Keep it low to avoid overloading the site or being banned.">
            Per minute <input type="number" min={1} max={120} value={perMinute} onChange={(e) => setPerMinute(+e.target.value)} />
          </label>
          <label title="Load pages in the browser connected over CDP (TINYLM_CDP_URL), for sites that need JavaScript">
            <input type="checkbox" checked={useBrowser} onChange={(e) => setUseBrowser(e.target.checked)} /> Browser (CDP)
          </label>
          <button type="submit" disabled={running || !urls.trim()}>Add website</button>
        </div>
      </form>
      {error && <p className="error">{error}</p>}
      {job && (
        <div className="crawl">
          <div>
            <strong>{running ? "Crawling" : job.state === "done" ? "Finished" : job.state === "cancelled" ? "Cancelled" : "Failed"}</strong>
            {" · "}
            {job.fetched} of up to {job.options.max_pages} pages · {job.indexed} added · {job.duplicates} already known ·{" "}
            {job.skipped} skipped · {job.failed} failed
            {running && <button className="cancel" onClick={() => client.cancelCrawl(job.id)}>Stop</button>}
          </div>
          {job.current && <div className="muted">Now: {job.current}</div>}
          {job.error && <div className="error">{job.error}</div>}
          {job.log.length > 0 && <pre className="muted">{job.log.slice(-8).join("\n")}</pre>}
        </div>
      )}
    </div>
  );
}
