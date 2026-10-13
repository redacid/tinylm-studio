import { ReactNode, useEffect, useState } from "react";
import {
  EngineClient,
  SearchProviderName,
  SearchSettings,
  SearchSettingsChanges,
} from "./api";

export const PROVIDER_TITLES: Record<SearchProviderName, string> = {
  searxng: "SearXNG",
  brave: "Brave Search",
  tavily: "Tavily",
  exa: "Exa",
  perplexity: "Perplexity",
  serper: "Serper (Google)",
};

const PROVIDERS: {
  id: SearchProviderName;
  about: string;
  link?: [string, string];
}[] = [
  {
    id: "searxng",
    about:
      "Self-hosted metasearch over Google, Bing, DuckDuckGo, YouTube and others. Free, no key; start it with make searxng.",
    link: ["https://docs.searxng.org/", "docs.searxng.org"],
  },
  {
    id: "brave",
    about: "Brave's own index, with web and video search. Needs an API key.",
    link: ["https://brave.com/search/api/", "brave.com/search/api"],
  },
  {
    id: "tavily",
    about:
      "Search made for AI applications; filters domains itself. Web pages only: videos are searched on YouTube and Vimeo.",
    link: ["https://app.tavily.com/", "app.tavily.com"],
  },
  {
    id: "exa",
    about:
      "Search by meaning, made for AI applications; filters domains itself. Web pages only: videos are searched on YouTube and Vimeo.",
    link: ["https://dashboard.exa.ai/", "dashboard.exa.ai"],
  },
  {
    id: "perplexity",
    about:
      "Perplexity's Search API: ranked pages with snippets; filters up to 20 domains itself. Web pages only.",
    link: ["https://docs.perplexity.ai/", "docs.perplexity.ai"],
  },
  {
    id: "serper",
    about:
      "Google's results through serper.dev, with web and video search. Needs an API key.",
    link: ["https://serper.dev/", "serper.dev"],
  },
];

type Draft = {
  provider: SearchProviderName;
  include: string;
  exclude: string;
  searxng_url: string;
  tavily_depth: SearchSettings["tavily_depth"];
  tavily_topic: SearchSettings["tavily_topic"];
  exa_type: SearchSettings["exa_type"];
  serper_country: string;
  /** New keys typed in; "" removes a saved key. Keys left out are kept. */
  keys: Partial<Record<Exclude<SearchProviderName, "searxng">, string>>;
};

function draftOf(s: SearchSettings): Draft {
  return {
    provider: s.provider,
    include: s.include_domains.join("\n"),
    exclude: s.exclude_domains.join("\n"),
    searxng_url: s.searxng_url,
    tavily_depth: s.tavily_depth,
    tavily_topic: s.tavily_topic,
    exa_type: s.exa_type,
    serper_country: s.serper_country ?? "",
    keys: {},
  };
}

const lines = (text: string) =>
  text
    .split(/[\n,]/)
    .map((l) => l.trim())
    .filter(Boolean);

/** The engine's search settings: provider, its options and key, and the domain filter. */
export function SearchSettingsForm({
  client,
}: {
  client: EngineClient | null;
}) {
  const [saved, setSaved] = useState<SearchSettings | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!client) return;
    client.searchSettings().then(
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
  const changes: SearchSettingsChanges = {
    provider: draft.provider,
    include_domains: lines(draft.include),
    exclude_domains: lines(draft.exclude),
    searxng_url: draft.searxng_url.trim(),
    tavily_depth: draft.tavily_depth,
    tavily_topic: draft.tavily_topic,
    exa_type: draft.exa_type,
    serper_country: draft.serper_country.trim().toLowerCase() || null,
    ...Object.fromEntries(
      Object.entries(draft.keys).map(([id, key]) => [`${id}_api_key`, key]),
    ),
  };
  const dirty = JSON.stringify(draftOf(saved)) !== JSON.stringify({ ...draft });

  async function save(): Promise<boolean> {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const next = await client!.saveSearchSettings(changes);
      setSaved(next);
      setDraft(draftOf(next));
      setNotice("Saved.");
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function test() {
    if (dirty && !(await save())) return;
    setBusy(true);
    setError(null);
    setNotice("Searching for “tinylm”…");
    try {
      const outcome = await client!.search("tinylm", "web", "", 5);
      const hidden = outcome.hidden
        ? `, ${outcome.hidden} hidden by the domain filter`
        : "";
      setNotice(
        `${PROVIDER_TITLES[outcome.provider]} works: ${outcome.data.length} results${hidden}.`,
      );
    } catch (e) {
      setNotice(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const p = draft.provider;
  return (
    <div className="settings-form">
      <h3>Search settings</h3>
      <p className="muted">
        Used by the search on the Data tab and by <code>make search</code>. Kept
        by the engine, so they also apply in a cluster;
        <code> TINYLM_*</code> variables give the defaults.
      </p>

      <h4>Provider</h4>
      <div className="provider-list">
        {PROVIDERS.map((provider) => (
          <label
            key={provider.id}
            className={`provider ${p === provider.id ? "active" : ""}`}
          >
            <input
              type="radio"
              name="provider"
              checked={p === provider.id}
              onChange={() => set({ provider: provider.id })}
            />
            <span>
              <strong>{PROVIDER_TITLES[provider.id]}</strong>
              {provider.id !== "searxng" && (
                <KeyBadge set={keySet(saved, draft, provider.id)} />
              )}
              <small className="muted">
                {provider.about}{" "}
                {provider.link && (
                  <a href={provider.link[0]} target="_blank" rel="noreferrer">
                    {provider.link[1]}
                  </a>
                )}
              </small>
            </span>
          </label>
        ))}
      </div>

      <h4>{PROVIDER_TITLES[p]}</h4>
      {p === "searxng" ? (
        <Field
          label="Address"
          hint="Where SearXNG answers; its JSON output must be on (see the user guide)."
        >
          <input
            value={draft.searxng_url}
            onChange={(e) => set({ searxng_url: e.target.value })}
          />
        </Field>
      ) : (
        <KeyField
          key={p}
          isSet={saved[`${p}_api_key_set`]}
          value={draft.keys[p]}
          onChange={(key) => set({ keys: { ...draft.keys, [p]: key } })}
          envName={`TINYLM_${p.toUpperCase()}_API_KEY`}
        />
      )}
      {p === "tavily" && (
        <div className="settings-grid">
          <Field
            label="Search depth"
            hint="Advanced finds more relevant passages and costs twice the credits."
          >
            <select
              value={draft.tavily_depth}
              onChange={(e) =>
                set({ tavily_depth: e.target.value as Draft["tavily_depth"] })
              }
            >
              <option value="basic">Basic</option>
              <option value="advanced">Advanced</option>
            </select>
          </Field>
          <Field label="Topic" hint="News looks at recent news sources.">
            <select
              value={draft.tavily_topic}
              onChange={(e) =>
                set({ tavily_topic: e.target.value as Draft["tavily_topic"] })
              }
            >
              <option value="general">General</option>
              <option value="news">News</option>
            </select>
          </Field>
        </div>
      )}
      {p === "exa" && (
        <Field
          label="Search type"
          hint="Auto picks the best method; fast and instant answer sooner with less depth."
        >
          <select
            value={draft.exa_type}
            onChange={(e) =>
              set({ exa_type: e.target.value as Draft["exa_type"] })
            }
          >
            <option value="auto">Auto</option>
            <option value="fast">Fast</option>
            <option value="instant">Instant</option>
          </select>
        </Field>
      )}
      {p === "serper" && (
        <Field
          label="Country"
          hint="Two-letter code of Google's country, e.g. ua or us; empty for any."
        >
          <input
            value={draft.serper_country}
            maxLength={2}
            onChange={(e) => set({ serper_country: e.target.value })}
          />
        </Field>
      )}

      <h4>Sites</h4>
      <div className="settings-grid">
        <Field
          label="Only these sites"
          hint="One domain or pattern per line; subdomains are included, * is any characters (*.ua, *.google.*). Empty: any site."
        >
          <textarea
            rows={5}
            placeholder={"e.g. ign.com\n*.ua"}
            value={draft.include}
            onChange={(e) => set({ include: e.target.value })}
          />
        </Field>
        <Field
          label="Never these sites"
          hint="One domain or pattern per line; subdomains are excluded too, * is any characters. Empty: none."
        >
          <textarea
            rows={5}
            placeholder={"e.g. pinterest.com\n*.ru"}
            value={draft.exclude}
            onChange={(e) => set({ exclude: e.target.value })}
          />
        </Field>
      </div>
      <p className="muted">
        Tavily, Exa and Perplexity filter by site themselves. With SearXNG,
        Brave and Serper the search asks for <code>site:</code> results of each
        listed site (up to 5; <code>*.ua</code> as <code>site:.ua</code>) and
        drops the others here, fetching more pages when the filter leaves too
        few. Patterns a provider cannot take, like <code>*.google.*</code>, are
        applied here only.
      </p>

      {error && <p className="error">{error}</p>}
      {notice && <p className="notice">{notice}</p>}
      <div className="buttons-row">
        <button onClick={save} disabled={busy || !dirty}>
          Save
        </button>
        <button
          onClick={test}
          disabled={busy}
          title="Save, then search for a word to check the provider"
        >
          Test search
        </button>
        {dirty && (
          <button onClick={() => setDraft(draftOf(saved))} disabled={busy}>
            Discard changes
          </button>
        )}
      </div>
    </div>
  );
}

function keySet(
  saved: SearchSettings,
  draft: Draft,
  id: Exclude<SearchProviderName, "searxng">,
): boolean {
  const typed = draft.keys[id];
  return typed === undefined ? saved[`${id}_api_key_set`] : typed !== "";
}

function KeyBadge({ set }: { set: boolean }) {
  return (
    <span className={`badge ${set ? "done" : ""}`}>
      {set ? "key set" : "no key"}
    </span>
  );
}

function KeyField({
  isSet,
  value,
  onChange,
  envName,
}: {
  isSet: boolean;
  value: string | undefined;
  onChange: (key: string | undefined) => void;
  envName: string;
}) {
  const removing = value === "";
  return (
    <Field
      label="API key"
      hint={`Kept by the engine and never shown again. ${envName} sets one without the app.`}
    >
      <span className="key-row">
        <input
          type="password"
          autoComplete="off"
          placeholder={
            removing
              ? "will be removed"
              : isSet
                ? "set: type to replace"
                : "paste the key"
          }
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value || undefined)}
        />
        {isSet && !removing && (
          <button onClick={() => onChange("")}>Remove</button>
        )}
        {removing && <button onClick={() => onChange(undefined)}>Keep</button>}
      </span>
    </Field>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: ReactNode;
}) {
  return (
    <label className="settings-field">
      <span>{label}</span>
      {children}
      <small className="muted">{hint}</small>
    </label>
  );
}
