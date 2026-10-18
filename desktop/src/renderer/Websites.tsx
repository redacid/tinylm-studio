import { useCallback, useEffect, useRef, useState } from "react";
import { CrawlJob, EngineClient, Website } from "./api";
import { useTabActive } from "./tabActive";
import { Icon } from "./Icon";
import { BusySpinner, useBusy } from "./Busy";

type Draft = {
  name: string;
  urls: string;
  max_pages: number;
  max_depth: number;
  per_minute: number;
  use_browser: boolean;
  only_under_start: boolean;
  /** Link patterns never followed, one per line. */
  exclude: string;
};

const NEW_SITE: Draft = {
  name: "",
  urls: "",
  max_pages: 50,
  max_depth: 2,
  per_minute: 20,
  use_browser: false,
  only_under_start: false,
  exclude: "",
};

/**
 * Saved websites of a collection. Each can be crawled again: pages downloaded before are skipped
 * and not counted, so every run adds up to "Pages" new pages until the whole site is covered.
 */
export function Websites({
  client,
  collection,
  onIndexed,
  savedVersion,
}: {
  client: EngineClient;
  collection: string;
  onIndexed: () => void;
  /** Changes when websites were saved elsewhere (from search results), to reload the list. */
  savedVersion: number;
}) {
  const [websites, setWebsites] = useState<Website[]>([]);
  const [jobs, setJobs] = useState<CrawlJob[]>([]);
  // The website being edited: its id, "new" for a new one, or null when the form is closed.
  const [editing, setEditing] = useState<number | "new" | null>(null);
  const [draft, setDraft] = useState<Draft>(NEW_SITE);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const seen = useRef(new Map<string, { indexed: number; state: string }>());

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  const loading = useBusy();
  const { run: load } = loading;
  const refreshWebsites = useCallback(async () => {
    setWebsites(await load("Loading websites…", () => client.websites(collection)));
  }, [client, collection, load]);

  useEffect(() => {
    setEditing(null);
    setConfirmDelete(null);
    setError(null);
    refreshWebsites().catch(report);
  }, [refreshWebsites]);

  useEffect(() => {
    if (savedVersion > 0) refreshWebsites().catch(report);
  }, [refreshWebsites, savedVersion]);

  // Poll crawls into this collection; reload documents as pages arrive and websites when one ends.
  const active = useTabActive();
  useEffect(() => {
    if (!active) return;
    let stopped = false;
    async function poll() {
      const list = (await client.crawls()).filter((j) => j.collection === collection);
      if (stopped) return;
      let added = false;
      let finished = false;
      for (const job of list) {
        const before = seen.current.get(job.id);
        if (before && before.indexed !== job.indexed) added = true;
        if (before?.state === "running" && job.state !== "running") finished = true;
        seen.current.set(job.id, { indexed: job.indexed, state: job.state });
      }
      setJobs(list);
      if (added || finished) onIndexed();
      if (finished) refreshWebsites().catch(() => {});
    }
    poll().catch(() => {});
    const timer = setInterval(() => poll().catch(() => {}), 1500);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [client, collection, onIndexed, refreshWebsites, active]);

  async function edit(website: Website | null) {
    setError(null);
    // A new website starts with the defaults from Settings > Crawler.
    const defaults = website ? null : await client.crawlerSettings().catch(() => null);
    setEditing(website ? website.id : "new");
    setDraft(
      website
        ? {
            name: website.name,
            urls: website.urls.join("\n"),
            max_pages: website.max_pages,
            max_depth: website.max_depth,
            per_minute: website.per_minute,
            use_browser: website.use_browser,
            only_under_start: website.only_under_start,
            exclude: website.exclude.join("\n"),
          }
        : defaults
          ? {
              ...NEW_SITE,
              max_pages: defaults.max_pages,
              max_depth: defaults.max_depth,
              per_minute: defaults.per_minute,
              use_browser: defaults.use_browser,
            }
          : NEW_SITE,
    );
  }

  async function save(event: { preventDefault(): void }, crawl: boolean) {
    event.preventDefault();
    const settings = {
      ...draft,
      urls: draft.urls.split(/\s+/).filter(Boolean),
      exclude: draft.exclude
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean),
    };
    if (settings.urls.length === 0) return;
    setError(null);
    try {
      const website =
        editing === "new" || editing === null
          ? await client.addWebsite(collection, settings)
          : await client.updateWebsite(editing, settings);
      setEditing(null);
      await refreshWebsites();
      if (crawl) await start(website.id);
    } catch (e) {
      report(e);
    }
  }

  async function start(id: number) {
    setError(null);
    try {
      const job = await client.crawlWebsite(id);
      setJobs((list) => [...list, job]);
    } catch (e) {
      report(e);
    }
  }

  async function remove(id: number) {
    setConfirmDelete(null);
    await client.deleteWebsite(id).catch(report);
    if (editing === id) setEditing(null);
    await refreshWebsites().catch(report);
  }

  const latest = (id: number) => jobs.filter((j) => j.website_id === id).pop();
  const set = (patch: Partial<Draft>) => setDraft((d) => ({ ...d, ...patch }));

  return (
    <div className="websites">
      <div className="websites-head">
        <strong>Websites</strong>
        {editing === null && <button onClick={() => edit(null)}><Icon name="world-plus" />Add website</button>}
      </div>
      {editing !== null && (
        <form onSubmit={(e) => save(e, true)}>
          <input placeholder="Name (optional)" value={draft.name} onChange={(e) => set({ name: e.target.value })} />
          <textarea
            placeholder={"Website addresses, one per line\nhttps://example.com/reviews/"}
            value={draft.urls}
            onChange={(e) => set({ urls: e.target.value })}
          />
          <div className="crawl-options">
            <label title="New pages to download per run. Pages downloaded before are skipped and not counted.">
              Pages <input type="number" min={1} value={draft.max_pages} onChange={(e) => set({ max_pages: +e.target.value })} />
            </label>
            <label title="How many links away from the start pages to go">
              Depth <input type="number" min={0} max={100} value={draft.max_depth} onChange={(e) => set({ max_depth: +e.target.value })} />
            </label>
            <label title="Requests per minute to each site. Keep it low to avoid overloading the site or being banned.">
              Per minute <input type="number" min={1} max={120} value={draft.per_minute} onChange={(e) => set({ per_minute: +e.target.value })} />
            </label>
            <label title="Load pages in the browser connected over CDP (its address is in Settings > Crawler), for sites that need JavaScript">
              <input type="checkbox" checked={draft.use_browser} onChange={(e) => set({ use_browser: e.target.checked })} /> Browser (CDP)
            </label>
            <label title="Follow links only inside the folders of the addresses above: from example.com/guides/ only to example.com/guides/…, not the rest of the site. Other sites and subdomains are never followed either way.">
              <input
                type="checkbox"
                checked={draft.only_under_start}
                onChange={(e) => set({ only_under_start: e.target.checked })}
              />{" "}
              Only under these addresses
            </label>
            <label
              className="exclude"
              title="One per line. A part of the address (/tag/, ?sort=, /login) matches anywhere in it; with * the pattern covers the whole address (*/forum/*, *.pdf). Letter case is ignored. The start addresses are always fetched."
            >
              Skip links matching
              <textarea
                placeholder={"/tag/\n?sort=\n*/forum/*"}
                value={draft.exclude}
                onChange={(e) => set({ exclude: e.target.value })}
              />
            </label>
            <span className="buttons">
              <button type="button" onClick={() => setEditing(null)}><Icon name="x" />Cancel</button>
              <button type="button" onClick={(e) => save(e, false)} disabled={!draft.urls.trim()}><Icon name="device-floppy" />Save</button>
              <button type="submit" disabled={!draft.urls.trim()}><Icon name="spider" />Save and crawl</button>
            </span>
          </div>
        </form>
      )}
      {error && <p className="error">{error}</p>}
      <BusySpinner busy={loading} />
      {websites.length === 0 && editing === null && !loading.busy && (
        <p className="muted">Save a website to crawl it into this collection, now and again later.</p>
      )}
      {websites.map((w) => {
        const job = latest(w.id);
        const running = job?.state === "running";
        return (
          <div key={w.id} className="website">
            <div className="website-row">
              <div className="website-name">
                <strong>{w.name}</strong>
                <div className="muted">
                  {w.urls[0]}
                  {w.urls.length > 1 && ` and ${w.urls.length - 1} more`}
                </div>
              </div>
              <div className="muted">
                {w.max_pages} new pages · depth {w.max_depth} · {w.per_minute}/min{w.use_browser && " · browser"}
                {w.only_under_start && " · only under these addresses"}
                {w.exclude.length > 0 && ` · skips ${w.exclude.join(", ")}`}
                <div>{w.last_crawled_at ? `Last crawled ${new Date(w.last_crawled_at).toLocaleString()}` : "Not crawled yet"}</div>
              </div>
              <div className="buttons">
                {running ? (
                  <button onClick={() => client.cancelCrawl(job.id)}><Icon name="player-stop" />Stop</button>
                ) : (
                  <button onClick={() => start(w.id)}><Icon name="spider" />Crawl</button>
                )}
                <button onClick={() => edit(w)} disabled={running}><Icon name="pencil" />Edit</button>
                {confirmDelete === w.id ? (
                  <>
                    <button className="danger" onClick={() => remove(w.id)}><Icon name="trash" />Delete</button>
                    <button onClick={() => setConfirmDelete(null)}><Icon name="x" />Cancel</button>
                  </>
                ) : (
                  <button onClick={() => setConfirmDelete(w.id)} disabled={running}><Icon name="trash" />Delete</button>
                )}
              </div>
            </div>
            {confirmDelete === w.id && (
              <div className="muted">Delete the saved website? Pages already added stay in the collection.</div>
            )}
            {job && <CrawlProgress job={job} />}
          </div>
        );
      })}
    </div>
  );
}

export function CrawlProgress({ job }: { job: CrawlJob }) {
  const running = job.state === "running";
  const state = running ? "Crawling" : job.state === "done" ? "Finished" : job.state === "cancelled" ? "Stopped" : "Failed";
  return (
    <div className="crawl">
      <div>
        <strong>{state}</strong> · {job.fetched} of {job.options.max_pages} new pages · {job.indexed} added ·{" "}
        {job.known} downloaded before · {job.duplicates} duplicates · {job.skipped} skipped · {job.failed} failed
      </div>
      {job.current && <div className="muted">Now: {job.current}</div>}
      {job.error && <div className="error">{job.error}</div>}
      {running && job.log.length > 0 && <pre className="muted">{job.log.slice(-6).join("\n")}</pre>}
    </div>
  );
}
