import { useEffect, useState } from "react";
import { CrawlerSettings, EngineClient } from "./api";
import { Field } from "./SearchSettings";
import { Icon } from "./Icon";

type Draft = Record<keyof CrawlerSettings, string | boolean>;

const draftOf = (s: CrawlerSettings): Draft => ({
  respect_robots: s.respect_robots,
  per_minute: String(s.per_minute),
  max_pages: String(s.max_pages),
  max_depth: String(s.max_depth),
  use_browser: s.use_browser,
  user_agent: s.user_agent,
  cdp_url: s.cdp_url,
});

/** The engine's crawler settings: robots.txt, request rate, defaults for new crawls, fetching. */
export function CrawlerSettingsForm({
  client,
}: {
  client: EngineClient | null;
}) {
  const [saved, setSaved] = useState<CrawlerSettings | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!client) return;
    client.crawlerSettings().then(
      (s) => {
        setSaved(s);
        setDraft(draftOf(s));
      },
      (e) => setError(String(e)),
    );
  }, [client]);

  if (!client)
    return (
      <p className="muted">
        These settings are kept by the engine; waiting for it to start…
      </p>
    );
  if (!draft || !saved)
    return error ? (
      <p className="error">{error}</p>
    ) : (
      <p className="muted">Loading…</p>
    );

  const set = (change: Partial<Draft>) => {
    setDraft({ ...draft, ...change });
    setNotice(null);
  };
  const dirty = JSON.stringify(draftOf(saved)) !== JSON.stringify(draft);

  async function save() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const values: CrawlerSettings = {
        respect_robots: draft!.respect_robots as boolean,
        per_minute: Number(draft!.per_minute),
        max_pages: Number(draft!.max_pages),
        max_depth: Number(draft!.max_depth),
        use_browser: draft!.use_browser as boolean,
        user_agent: String(draft!.user_agent),
        cdp_url: String(draft!.cdp_url),
      };
      // Only what changed, so the other settings keep following the engine's TINYLM_* defaults.
      const changes = Object.fromEntries(
        Object.entries(values).filter(
          ([name, value]) => saved![name as keyof CrawlerSettings] !== value,
        ),
      );
      const next = await client!.saveCrawlerSettings(changes);
      setSaved(next);
      setDraft(draftOf(next));
      setNotice("Saved. Crawls started from now on use these settings.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const text = (
    name: keyof CrawlerSettings,
    type = "text",
    min?: number,
    max?: number,
  ) => (
    <input
      type={type}
      min={min}
      max={max}
      value={draft[name] as string}
      onChange={(e) => set({ [name]: e.target.value })}
    />
  );

  return (
    <div className="settings-form">
      <h3>Crawler settings</h3>
      <p className="muted">
        Used when pages and websites are added on the Data tab and by{" "}
        <code>make crawl</code>. Kept by the engine, so they also apply in a
        cluster; <code>TINYLM_*</code> variables give the defaults.
      </p>

      <h4>robots.txt</h4>
      <div className="provider-list">
        <label className={`provider ${draft.respect_robots ? "active" : ""}`}>
          <input
            type="radio"
            name="robots"
            checked={!!draft.respect_robots}
            onChange={() => set({ respect_robots: true })}
          />
          <span>
            <strong>Follow robots.txt</strong>
            <small className="muted">
              Each site's robots.txt is read first: pages it disallows are
              skipped, and its Crawl-delay slows the crawl down. Recommended.
            </small>
          </span>
        </label>
        <label className={`provider ${draft.respect_robots ? "" : "active"}`}>
          <input
            type="radio"
            name="robots"
            checked={!draft.respect_robots}
            onChange={() => set({ respect_robots: false })}
          />
          <span>
            <strong>Ignore robots.txt</strong>
            <small className="muted">
              robots.txt is not read; every page is downloaded. Only for sites
              you may crawl anyway, such as your own: sites may ban crawlers
              that ignore it.
            </small>
          </span>
        </label>
      </div>

      <h4>New crawls</h4>
      <div className="settings-grid">
        <Field
          label="Requests per minute"
          hint="To each site, from 1 to 120. Keep it low to avoid overloading or being banned."
        >
          {text("per_minute", "number", 1, 120)}
        </Field>
        <Field
          label="Pages"
          hint="New pages a website downloads per run; pages downloaded before are skipped."
        >
          {text("max_pages", "number", 1, 100000)}
        </Field>
        <Field
          label="Depth"
          hint="How many links away from the start pages to go; 0 is only the start pages."
        >
          {text("max_depth", "number", 0, 100)}
        </Field>
      </div>
      <p className="muted">
        A new website starts with these values; each saved website keeps its own
        after that.
      </p>

      <h4>Fetching</h4>
      <div className="provider-list">
        <label className={`provider ${draft.use_browser ? "" : "active"}`}>
          <input
            type="radio"
            name="fetch"
            checked={!draft.use_browser}
            onChange={() => set({ use_browser: false })}
          />
          <span>
            <strong>Plain HTTP</strong>
            <small className="muted">Fast, and enough for most sites.</small>
          </span>
        </label>
        <label className={`provider ${draft.use_browser ? "active" : ""}`}>
          <input
            type="radio"
            name="fetch"
            checked={!!draft.use_browser}
            onChange={() => set({ use_browser: true })}
          />
          <span>
            <strong>Browser (CDP)</strong>
            <small className="muted">
              Pages load in the browser at the address below, for sites that
              need JavaScript or block simple clients. Slower.
            </small>
          </span>
        </label>
      </div>
      <p className="muted">
        Used for pages added from search results and as the start for new
        websites; a saved website uses its own Browser (CDP) box.
      </p>
      <Field
        label="User agent"
        hint="How the crawler introduces itself to sites, and the name robots.txt rules are read for."
      >
        {text("user_agent")}
      </Field>
      <Field
        label="Browser address (CDP)"
        hint="Chrome's DevTools endpoint, used only when pages load through the browser. make chrome starts one in Docker."
      >
        {text("cdp_url")}
      </Field>

      {error && <p className="error">{error}</p>}
      {notice && <p className="notice">{notice}</p>}
      <div className="buttons-row">
        <button onClick={save} disabled={busy || !dirty}>
          <Icon name="device-floppy" />Save
        </button>
        {dirty && (
          <button onClick={() => setDraft(draftOf(saved))} disabled={busy}>
            <Icon name="arrow-back-up" />Discard changes
          </button>
        )}
      </div>
    </div>
  );
}
