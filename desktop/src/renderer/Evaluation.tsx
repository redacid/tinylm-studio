import { FormEvent, useCallback, useEffect, useState } from "react";
import { Dataset, EngineClient, Evaluation, EvaluationOptions, Teacher, TrainedModel } from "./api";
import { useTabActive } from "./tabActive";
import { Icon } from "./Icon";
import { BusySpinner, useBusy } from "./Busy";

const STATE_LABELS: Record<Evaluation["state"], string> = {
  running: "Running",
  done: "Finished",
  cancelled: "Stopped",
  error: "Failed",
  interrupted: "Interrupted",
};
const MAX_MODELS = 4;

/** Evaluation: compare models on the held-out examples of a dataset. */
export function EvaluationScreen({ client }: { client: EngineClient }) {
  const [runs, setRuns] = useState<Evaluation[]>([]);
  // A run's id, or "new" for the form.
  const [selected, setSelected] = useState<string>("new");
  const [error, setError] = useState<string | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  const refresh = useCallback(async () => setRuns((await client.evaluations()).reverse()), [client]);
  const loading = useBusy();
  const { run: load } = loading;

  const active = useTabActive();
  useEffect(() => {
    if (!active) return;
    load("Loading evaluations…", refresh).catch(report);
    const timer = setInterval(() => refresh().catch(() => {}), 3000);
    return () => clearInterval(timer);
  }, [refresh, active, load]);

  return (
    <div className="data">
      <aside>
        <strong>Evaluations</strong>
        <BusySpinner busy={loading} />
        <button className={selected === "new" ? "active" : ""} onClick={() => setSelected("new")}>
          <Icon name="plus" />New evaluation
        </button>
        {runs.map((run) => (
          <button key={run.id} className={run.id === selected ? "active" : ""} onClick={() => setSelected(run.id)}>
            {run.options.dataset} <span className={`badge ${run.state}`}>{STATE_LABELS[run.state]}</span>
            <div className="muted">
              {new Date(run.created_at).toLocaleString()} · {run.options.models.join(", ")}
            </div>
          </button>
        ))}
      </aside>
      <section>
        {error && <p className="error">{error}</p>}
        {selected === "new" ? (
          <EvaluationForm
            client={client}
            onStarted={(run) => {
              setSelected(run.id);
              refresh().catch(report);
            }}
          />
        ) : (
          <Report
            key={selected}
            client={client}
            id={selected}
            onDeleted={() => {
              setSelected("new");
              refresh().catch(report);
            }}
            onFinished={() => refresh().catch(report)}
          />
        )}
      </section>
    </div>
  );
}

function EvaluationForm({ client, onStarted }: { client: EngineClient; onStarted: (run: Evaluation) => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [trained, setTrained] = useState<TrainedModel[]>([]);
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [options, setOptions] = useState<EvaluationOptions>({
    dataset: "",
    models: [],
    judge: null,
    limit: 50,
    max_tokens: 512,
  });
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<EvaluationOptions>) => setOptions((o) => ({ ...o, ...patch }));

  const active = useTabActive();
  useEffect(() => {
    if (!active) return;
    Promise.all([client.datasets(), client.installedModels(), client.trainedModels(), client.teachers()])
      .then(([d, m, t, j]) => {
        setDatasets(d);
        setModels(m);
        setTrained(t);
        setTeachers(j);
        // Shown again: keep what was chosen while the dataset still exists.
        setOptions((o) => {
          if (d.some((x) => x.name === o.dataset)) return o;
          const dataset = d.find((x) => x.eval > 0)?.name ?? d[0]?.name ?? "";
          return { ...o, dataset, models: suggested(dataset, t, m) };
        });
      })
      .catch((e) => setError(String(e)));
  }, [client, active]);

  function toggle(model: string) {
    set({
      models: options.models.includes(model) ? options.models.filter((m) => m !== model) : [...options.models, model],
    });
  }

  async function start(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      onStarted(await client.startEvaluation(options));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const dataset = datasets.find((d) => d.name === options.dataset);
  const count = Math.min(dataset?.eval ?? 0, options.limit);
  const saved = teachers.filter((t) => t.kind !== "local");
  const local = teachers.filter((t) => t.kind === "local");
  return (
    <form className="training-form" onSubmit={start}>
      <strong>Compare models</strong>
      <p className="muted">
        Each model answers the held-out questions of a dataset (examples it was not trained on). Answers are compared
        with the dataset's answers by similarity and, if you choose a judge, scored from 1 to 5 by a teacher model.
      </p>
      <div className="crawl-options">
        <label>
          Dataset{" "}
          <select
            value={options.dataset}
            onChange={(e) => set({ dataset: e.target.value, models: suggested(e.target.value, trained, models) })}
          >
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.eval} held out)
              </option>
            ))}
          </select>
        </label>
        <label title="Held-out examples to use; fewer is faster">
          Examples
          <input type="number" min={1} value={options.limit} onChange={(e) => set({ limit: +e.target.value })} />
        </label>
        <label title="The longest answer a model may write, in tokens">
          Max answer tokens
          <input type="number" min={16} value={options.max_tokens} onChange={(e) => set({ max_tokens: +e.target.value })} />
        </label>
      </div>
      {dataset && dataset.eval === 0 && (
        <p className="error">This dataset has no held-out examples: set its evaluation share on the Datasets screen.</p>
      )}
      <div>
        Models (up to {MAX_MODELS}):
        <div className="model-choices">
          {models.map((m) => (
            <label key={m}>
              <input
                type="checkbox"
                checked={options.models.includes(m)}
                disabled={!options.models.includes(m) && options.models.length >= MAX_MODELS}
                onChange={() => toggle(m)}
              />{" "}
              {m}
              {trained.some((t) => t.id === m) && <span className="muted"> fine-tuned</span>}
            </label>
          ))}
        </div>
      </div>
      <label>
        Judge{" "}
        <select value={options.judge ?? ""} onChange={(e) => set({ judge: e.target.value || null })}>
          <option value="">None: similarity only</option>
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
      <p className="muted">
        {count} questions × {options.models.length} models = {count * options.models.length} answers
        {options.judge ? `, plus ${count} judge requests` : ""}. Models answer one after another on this engine; a
        judge is a teacher from the Models screen or an installed model.
        {options.judge?.startsWith("local:") &&
          " An installed judge runs on this engine after all answers are written; pick one larger than the models it scores, it is only as good as its own answers."}
      </p>
      {error && <p className="error">{error}</p>}
      <div className="buttons">
        <button type="submit" disabled={!count || options.models.length === 0}><Icon name="player-play" />Start evaluation</button>
      </div>
    </form>
  );
}

/** The fine-tuned models of a dataset and their base models, the natural comparison. */
function suggested(dataset: string, trained: TrainedModel[], installed: string[]): string[] {
  const picks = trained.filter((t) => t.dataset === dataset).flatMap((t) => [t.base, t.id]);
  return [...new Set(picks)].filter((m) => installed.includes(m)).slice(0, MAX_MODELS);
}

function Report({
  client,
  id,
  onDeleted,
  onFinished,
}: {
  client: EngineClient;
  id: string;
  onDeleted: () => void;
  onFinished: () => void;
}) {
  const [run, setRun] = useState<Evaluation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let stopped = false;
    let wasRunning = false;
    const load = async () => {
      try {
        const next = await client.evaluation(id);
        if (stopped) return;
        setRun(next);
        if (next.state === "running") {
          wasRunning = true;
          timer = setTimeout(load, 1500);
        } else if (wasRunning) {
          onFinished();
        }
      } catch (e) {
        report(e);
      }
    };
    load();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [client, id]);

  if (!run) return error ? <p className="error">{error}</p> : null;
  const models = run.options.models;
  const judged = models.some((m) => run.summary[m]?.judged);
  const best = (key: "score" | "similarity") => Math.max(...models.map((m) => run.summary[m]?.[key] ?? -1));

  async function remove() {
    await client.deleteEvaluation(id).then(onDeleted, report);
  }

  return (
    <div className="evaluation">
      <div className="example-head">
        <strong>{run.options.dataset}</strong>
        <span className={`badge ${run.state}`}>{STATE_LABELS[run.state]}</span>
        <span className="muted">
          {new Date(run.created_at).toLocaleString()}
          {run.options.judge ? "" : " · similarity only"}
        </span>
        <span className="buttons">
          {run.state === "running" ? (
            <button onClick={() => client.stopEvaluation(id).catch(report)}><Icon name="player-stop" />Stop</button>
          ) : (
            <button onClick={remove}><Icon name="trash" />Delete</button>
          )}
        </span>
      </div>
      {run.state === "running" && (
        <>
          <div className="progress">
            <div style={{ width: `${run.total ? (100 * run.done) / run.total : 0}%` }} />
          </div>
          <span className="muted">
            {run.phase === "judging" ? "The judge is scoring the answers" : `${run.current ?? "A model"} is answering`} ·{" "}
            {run.done} of {run.total}
          </span>
        </>
      )}
      {run.error && <p className="error">{run.error}</p>}
      {error && <p className="error">{error}</p>}
      <table className="summary">
        <thead>
          <tr>
            <th>Model</th>
            {judged && <th title="The judge's average score, 1 to 5">Judge score</th>}
            <th title="Character n-gram similarity to the dataset's answers (chrF), 0 to 100">Similarity</th>
            <th>Seconds per answer</th>
            <th>Answer length</th>
            <th>Answers</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const s = run.summary[m];
            return (
              <tr key={m}>
                <td>{m}</td>
                {judged && (
                  <td className={s?.score != null && s.score === best("score") ? "best" : ""}>
                    {s?.score != null ? (
                      <>
                        {s.score.toFixed(2)} <Meter value={(s.score - 1) / 4} />
                      </>
                    ) : (
                      "–"
                    )}
                  </td>
                )}
                <td className={s?.similarity != null && s.similarity === best("similarity") ? "best" : ""}>
                  {s?.similarity != null ? (
                    <>
                      {s.similarity.toFixed(1)} <Meter value={s.similarity / 100} />
                    </>
                  ) : (
                    "–"
                  )}
                </td>
                <td>{s?.seconds ?? "–"}</td>
                <td>{s?.chars != null ? `${s.chars} chars` : "–"}</td>
                <td>{s?.answers ?? 0}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {(run.examples ?? []).map((example, i) => {
        const question = [...example.messages].reverse().find((m) => m.role === "user")?.content ?? "";
        return (
          <div key={i} className="example">
            <div className="turn user">
              <span className="role">question</span>
              <span className="text">{question}</span>
            </div>
            <div className="turn assistant">
              <span className="role">expected</span>
              <span className="text">{example.reference}</span>
            </div>
            {models
              .filter((m) => example.answers[m])
              .map((m) => {
                const a = example.answers[m];
                return (
                  <div key={m} className="turn answer">
                    <span className="role" title={m}>{m}</span>
                    <span className="text">
                      {a.text || <span className="muted">(empty)</span>}
                      <div className="muted">
                        {a.score != null && <span className={`score s${a.score}`}>{a.score}/5</span>}{" "}
                        {a.reason && `${a.reason} · `}similarity {a.similarity} · {a.seconds} s
                      </div>
                    </span>
                  </div>
                );
              })}
            {example.judge_error && <p className="error">Judge: {example.judge_error}</p>}
          </div>
        );
      })}
    </div>
  );
}

function Meter({ value }: { value: number }) {
  return (
    <span className="meter">
      <span style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </span>
  );
}
