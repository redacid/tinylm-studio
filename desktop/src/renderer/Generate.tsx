import { FormEvent, useEffect, useState } from "react";
import { Collection, EngineClient, GenerationJob, GenerationOptions, Served, Teacher, TeacherRequest } from "./api";
import { useTabActive } from "./tabActive";

const LANGUAGES = [
  ["uk", "Ukrainian"],
  ["ru", "Russian"],
  ["en", "English"],
] as const;

const DEFAULTS: GenerationOptions = {
  collection: "",
  teacher: "",
  chunks: 20,
  per_chunk: 3,
  languages: [],
  instructions: "",
  system_prompt: "",
  unanswerable: true,
  verify: true,
};
const SETTINGS_KEY = "generate-options";

function savedOptions(): GenerationOptions {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

/** Writes examples for a dataset from a collection's documents with a teacher model. */
export function GeneratePanel({
  client,
  dataset,
  onAdded,
  onClose,
}: {
  client: EngineClient;
  dataset: string;
  onAdded: () => void;
  onClose: () => void;
}) {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [options, setOptions] = useState<GenerationOptions>(savedOptions);
  const [unused, setUnused] = useState<number | null>(null);
  const [job, setJob] = useState<GenerationJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  const set = (patch: Partial<GenerationOptions>) => setOptions((o) => ({ ...o, ...patch }));

  const active = useTabActive();
  useEffect(() => {
    if (!active) return;
    Promise.all([client.collections(), client.teachers(), client.generations()])
      .then(([c, t, jobs]) => {
        setCollections(c);
        setTeachers(t);
        setOptions((o) => ({
          ...o,
          collection: c.some((x) => x.name === o.collection) ? o.collection : c[0]?.name ?? "",
          teacher: t.some((x) => x.ref === o.teacher) ? o.teacher : t[0]?.ref ?? "",
        }));
        // The latest run for this dataset, so a running one can be followed after switching pages.
        const mine = jobs.filter((j) => j.dataset === dataset);
        setJob(mine[mine.length - 1] ?? null);
      })
      .catch(report);
  }, [client, dataset, active]);

  const running = job?.state === "running";

  useEffect(() => {
    if (!options.collection) return setUnused(null);
    client.unusedChunks(dataset, options.collection).then(setUnused, () => setUnused(null));
  }, [client, dataset, options.collection, job?.state]);

  useEffect(() => {
    if (!job || !running || !active) return;
    let added = job.added;
    const timer = setInterval(async () => {
      try {
        const next = await client.generation(job.id);
        setJob(next);
        if (next.added !== added || next.state !== "running") onAdded();
        added = next.added;
      } catch (e) {
        report(e);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [client, job?.id, running, active]);

  async function start(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...options }));
    } catch {
      // Remembering the settings is only a convenience.
    }
    try {
      setJob(await client.generate(dataset, options));
    } catch (e) {
      report(e);
    }
  }

  function toggleLanguage(code: string) {
    set({
      languages: options.languages.includes(code)
        ? options.languages.filter((l) => l !== code)
        : [...options.languages, code],
    });
  }

  const saved = teachers.filter((t) => t.kind !== "local");
  const local = teachers.filter((t) => t.kind === "local");

  return (
    <div className="generate">
      <form className="training-form" onSubmit={start}>
        <div className="websites-head">
          <strong>Generate examples</strong>
          <button type="button" onClick={onClose}>Close</button>
        </div>
        <p className="muted">
          A larger teacher model reads documents of a collection and writes question and answer pairs from them. With
          checking on, it then drops the pairs the text does not support. Each run continues with documents not used yet.
        </p>
        <div className="crawl-options">
          <label>
            From collection{" "}
            <select value={options.collection} onChange={(e) => set({ collection: e.target.value })}>
              {collections.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </label>
          {unused !== null && <span className="muted">{unused} chunks not used yet</span>}
          <label>
            Teacher{" "}
            <select value={options.teacher} onChange={(e) => set({ teacher: e.target.value })}>
              {saved.length > 0 && (
                <optgroup label="API providers">
                  {saved.map((t) => (
                    <option key={t.ref} value={t.ref}>{t.name}</option>
                  ))}
                </optgroup>
              )}
              {local.length > 0 && (
                <optgroup label="Installed models">
                  {local.map((t) => (
                    <option key={t.ref} value={t.ref}>{t.name}</option>
                  ))}
                </optgroup>
              )}
            </select>
          </label>
        </div>
        {teachers.length === 0 && (
          <p className="muted">No teachers yet: add an API provider such as OpenRouter on the Models page, or install a model.</p>
        )}
        {collections.length === 0 && <p className="muted">Add documents to a collection on the Data page first.</p>}
        <div className="crawl-options">
          <label title="Chunks of text the teacher reads in this run">
            Chunks
            <input type="number" min={1} max={10000} value={options.chunks} onChange={(e) => set({ chunks: +e.target.value })} />
          </label>
          <label title="Question and answer pairs written from each chunk">
            Pairs per chunk
            <input type="number" min={1} max={20} value={options.per_chunk} onChange={(e) => set({ per_chunk: +e.target.value })} />
          </label>
          <span>
            Languages:{" "}
            {LANGUAGES.map(([code, name]) => (
              <label key={code}>
                <input type="checkbox" checked={options.languages.includes(code)} onChange={() => toggleLanguage(code)} /> {name}{" "}
              </label>
            ))}
            {options.languages.length === 0 && <span className="muted">same as each text</span>}
          </span>
        </div>
        <label>
          Answer style
          <textarea
            placeholder="e.g. Start with a one-line tip, then numbered steps. Mention the game's name."
            value={options.instructions}
            onChange={(e) => set({ instructions: e.target.value })}
          />
        </label>
        <label>
          System prompt of the examples
          <textarea
            placeholder="e.g. You are a helpful game guide."
            value={options.system_prompt}
            onChange={(e) => set({ system_prompt: e.target.value })}
          />
        </label>
        <div className="crawl-options">
          <label title="Teaches the model to say it does not know instead of inventing">
            <input type="checkbox" checked={options.unanswerable} onChange={(e) => set({ unanswerable: e.target.checked })} /> Add a
            question the text does not answer
          </label>
          <label title="One more request per chunk: the teacher checks each pair against the text">
            <input type="checkbox" checked={options.verify} onChange={(e) => set({ verify: e.target.checked })} /> Check pairs
            against the text
          </label>
          <span className="buttons">
            {running ? (
              <button type="button" onClick={() => client.cancelGeneration(job.id).catch(report)}>Stop</button>
            ) : (
              <button type="submit" disabled={!options.collection || !options.teacher || unused === 0}>Generate</button>
            )}
          </span>
        </div>
        <p className="muted">
          The examples of the dataset that you wrote or imported are shown to the teacher as samples of the wanted style.
        </p>
      </form>
      {error && <p className="error">{error}</p>}
      {job && <GenerationProgress job={job} />}
    </div>
  );
}

/** "deepseek/deepseek-chat via Chutes": the model and provider that answered. */
export function servedBy(served: Served): string {
  return [served.model, served.provider && served.provider !== "local" && `via ${served.provider}`]
    .filter(Boolean)
    .join(" ");
}

function GenerationProgress({ job }: { job: GenerationJob }) {
  const rejected = Object.entries(job.rejected);
  const share = job.chunks_total ? (100 * job.chunks_done) / job.chunks_total : 0;
  return (
    <div className="job">
      <div className="example-head">
        <span className={`badge ${job.state}`}>{job.state}</span>
        <span>
          {job.chunks_done} of {job.chunks_total} chunks · {job.generated} pairs written · {job.added} added
          {job.duplicates > 0 && ` · ${job.duplicates} already there`}
          {job.failed_chunks > 0 && ` · ${job.failed_chunks} chunks failed`}
        </span>
      </div>
      {job.state === "running" && (
        <div className="progress">
          <div style={{ width: `${share}%` }} />
        </div>
      )}
      {job.current && <span className="muted">Reading {job.current}</span>}
      {job.waiting != null && (
        <span className="muted">The provider's rate limit was reached; waiting {Math.round(job.waiting)} s.</span>
      )}
      {rejected.length > 0 && (
        <span className="muted">Dropped: {rejected.map(([reason, count]) => `${count} ${reason}`).join(", ")}</span>
      )}
      {job.error && <p className="error">{job.error}</p>}
      {job.requests.length > 0 && <TeacherRequests requests={job.requests} />}
      {job.state !== "running" && job.chunks_total === 0 && (
        <span className="muted">No chunks of this collection are left unused for this dataset.</span>
      )}
    </div>
  );
}

/** Each request to the teacher, newest first: who answered it and what came back. */
function TeacherRequests({ requests }: { requests: TeacherRequest[] }) {
  const [all, setAll] = useState(false);
  const shown = [...requests].reverse().slice(0, all ? undefined : 10);
  return (
    <div className="teacher-requests">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Document</th>
            <th>Request</th>
            <th>Model</th>
            <th>Provider</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={requests.length - i} className={r.ok === false ? "failed" : ""}>
              <td className="nowrap">{new Date(r.at).toLocaleTimeString()}</td>
              <td>{r.document}</td>
              <td className="nowrap">{r.step === "write" ? "write pairs" : "check pairs"}</td>
              <td>{r.model ?? "—"}</td>
              <td className="nowrap">{r.provider ?? "—"}</td>
              <td>{r.ok === null ? "waiting for the answer…" : r.result}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {requests.length > 10 && (
        <button type="button" onClick={() => setAll(!all)}>
          {all ? "Show the last 10" : `Show all ${requests.length}`}
        </button>
      )}
    </div>
  );
}
