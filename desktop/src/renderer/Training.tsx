import { FormEvent, useCallback, useEffect, useState } from "react";
import { Dataset, EngineClient, LossPoint, TrainingJob, TrainingSettings } from "./api";
import { useTabActive } from "./tabActive";

const DEFAULTS: Omit<TrainingSettings, "name" | "base" | "dataset"> = {
  epochs: 3,
  learning_rate: 2e-4,
  lora_rank: 16,
  max_length: 1024,
  batch_size: 1,
  gradient_accumulation: 4,
  device: "auto",
};

const STATE_LABELS: Record<TrainingJob["state"], string> = {
  running: "Training",
  stopping: "Stopping…",
  stopped: "Stopped",
  done: "Finished",
  error: "Failed",
  interrupted: "Interrupted",
};

/** Fine-tuning: start a LoRA run on a dataset and follow its loss; the result is a new model. */
export function Training({ client, onNeedModel }: { client: EngineClient; onNeedModel: () => void }) {
  const [bases, setBases] = useState<string[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [settings, setSettings] = useState<TrainingSettings>({ name: "", base: "", dataset: "", ...DEFAULTS });
  const [error, setError] = useState<string | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  const set = (patch: Partial<TrainingSettings>) => setSettings((s) => ({ ...s, ...patch }));

  const refreshJobs = useCallback(async () => setJobs((await client.trainingJobs()).reverse()), [client]);

  const active = useTabActive();
  useEffect(() => {
    if (!active) return;
    (async () => {
      const [models, sets] = await Promise.all([client.trainableModels(), client.datasets()]);
      setBases(models);
      setDatasets(sets);
      setSettings((s) => ({
        ...s,
        base: models.includes(s.base) ? s.base : models.find((m) => m.startsWith("qwen3-0.6b")) ?? models[0] ?? "",
        dataset: sets.some((d) => d.name === s.dataset) ? s.dataset : sets[0]?.name ?? "",
      }));
    })().catch(report);
    refreshJobs().catch(report);
    const timer = setInterval(() => refreshJobs().catch(() => {}), 3000);
    return () => clearInterval(timer);
  }, [client, refreshJobs, active]);

  async function start(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      await client.startTraining({ ...settings, name: settings.name.trim() });
      set({ name: "" });
      await refreshJobs();
    } catch (e) {
      report(e);
    }
  }

  const busy = jobs.some((j) => j.state === "running" || j.state === "stopping");
  const dataset = datasets.find((d) => d.name === settings.dataset);
  const steps = dataset
    ? Math.ceil(dataset.train / (settings.batch_size * settings.gradient_accumulation)) * settings.epochs
    : 0;

  return (
    <div className="training">
      <form className="training-form" onSubmit={start}>
        <strong>New fine-tuned model</strong>
        {bases.length === 0 ? (
          <p className="muted">
            Fine-tuning needs a model with PyTorch weights, such as qwen3-0.6b.{" "}
            <button type="button" onClick={onNeedModel}>Download one</button>
          </p>
        ) : datasets.length === 0 ? (
          <p className="muted">Create a dataset of examples on the Datasets page first.</p>
        ) : null}
        <div className="crawl-options">
          <label>
            Name <input placeholder="e.g. game-hints" value={settings.name} onChange={(e) => set({ name: e.target.value })} />
          </label>
          <label>
            Base model{" "}
            <select value={settings.base} onChange={(e) => set({ base: e.target.value })}>
              {bases.map((b) => <option key={b}>{b}</option>)}
            </select>
          </label>
          <label>
            Dataset{" "}
            <select value={settings.dataset} onChange={(e) => set({ dataset: e.target.value })}>
              {datasets.map((d) => <option key={d.name} value={d.name}>{d.name} ({d.train} examples)</option>)}
            </select>
          </label>
        </div>
        <div className="crawl-options">
          <label title="Passes over the training examples">
            Epochs <input type="number" min={0.1} step="any" value={settings.epochs} onChange={(e) => set({ epochs: +e.target.value })} />
          </label>
          <label title="How fast the adapter changes. 1e-4 to 3e-4 suits LoRA; too high makes the loss jump">
            Learning rate <input type="number" min={0} step="any" value={settings.learning_rate} onChange={(e) => set({ learning_rate: +e.target.value })} />
          </label>
          <label title="Size of the LoRA adapter: higher learns more but trains slower (8–32 is typical)">
            LoRA rank <input type="number" min={1} max={256} value={settings.lora_rank} onChange={(e) => set({ lora_rank: +e.target.value })} />
          </label>
          <label title="Longer examples are cut to this many tokens; shorter means faster and less memory">
            Max tokens <input type="number" min={32} value={settings.max_length} onChange={(e) => set({ max_length: +e.target.value })} />
          </label>
          <label title="Examples per step">
            Batch <input type="number" min={1} max={256} value={settings.batch_size} onChange={(e) => set({ batch_size: +e.target.value })} />
          </label>
          <label title="Steps whose gradients are added up before the model changes: a larger batch without more memory">
            Accumulate <input type="number" min={1} max={256} value={settings.gradient_accumulation} onChange={(e) => set({ gradient_accumulation: +e.target.value })} />
          </label>
          <label>
            Device{" "}
            <select value={settings.device} onChange={(e) => set({ device: e.target.value as TrainingSettings["device"] })}>
              <option value="auto">Auto (GPU if available)</option>
              <option value="cpu">CPU</option>
              <option value="cuda">NVIDIA GPU (CUDA)</option>
            </select>
          </label>
          <span className="buttons">
            {steps > 0 && <span className="muted">{steps} steps</span>}
            <button type="submit" disabled={busy || !settings.name.trim() || !settings.base || !dataset?.train}>
              Start training
            </button>
          </span>
        </div>
        {busy && <p className="muted">One run at a time: wait for the current run or stop it.</p>}
        {error && <p className="error">{error}</p>}
      </form>
      {jobs.length === 0 && <p className="muted">No training runs yet.</p>}
      {jobs.map((job) => (
        <JobCard key={job.id} client={client} job={job} onChange={() => refreshJobs().catch(report)} />
      ))}
    </div>
  );
}

function JobCard({ client, job, onChange }: { client: EngineClient; job: TrainingJob; onChange: () => void }) {
  const [history, setHistory] = useState<Map<number, LossPoint>>(new Map());
  const [live, setLive] = useState<Partial<TrainingJob>>({});
  const [log, setLog] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const going = job.state === "running" || job.state === "stopping";

  // The full loss history once, then the run's events while it is going.
  useEffect(() => {
    const abort = new AbortController();
    const merge = (points: LossPoint[]) =>
      setHistory((h) => {
        const next = new Map(h);
        for (const p of points) next.set(p.step, { ...next.get(p.step), ...p });
        return next;
      });
    client.trainingJob(job.id).then((full) => {
      merge(full.history ?? []);
      setLog(full.log ?? "");
    }, () => {});
    if (going) {
      client
        .trainingEvents(
          job.id,
          (event) => {
            if (event.type === "step") {
              merge([{ step: event.step, loss: event.loss }]);
              setLive((l) => ({ ...l, step: event.step, epoch: event.epoch, loss: event.loss }));
            } else if (event.type === "eval") {
              merge([{ step: event.step, eval_loss: event.eval_loss }]);
              setLive((l) => ({ ...l, eval_loss: event.eval_loss }));
            } else if (event.type === "start") {
              setLive((l) => ({ ...l, total_steps: event.total_steps }));
            } else if (event.type === "state" && event.state !== "running" && event.state !== "stopping") {
              onChange();
            }
          },
          abort.signal,
        )
        .catch(() => {});
    }
    return () => abort.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, job.id, going]);

  const current = { ...job, ...(going ? live : {}) };
  const total = current.total_steps ?? 0;
  const points = [...history.values()].sort((a, b) => a.step - b.step);
  const s = job.settings;

  return (
    <div className="job">
      <div className="website-row">
        <div className="website-name">
          <strong>{s.name}</strong>{" "}
          <span className={`badge ${job.state}`}>{STATE_LABELS[job.state]}</span>
          <div className="muted">
            {s.base} + {s.dataset} · {s.epochs} epochs · lr {s.learning_rate} · rank {s.lora_rank} · {s.max_length} tokens · batch{" "}
            {s.batch_size}
            {s.gradient_accumulation > 1 && `×${s.gradient_accumulation}`}
            {job.device && ` · ${job.device}`}
          </div>
        </div>
        <div className="buttons">
          {job.state === "running" && <button onClick={() => client.stopTraining(job.id).then(onChange)}>Stop</button>}
          {job.resumable && <button onClick={() => client.resumeTraining(job.id).then(onChange, (e) => setLog(String(e)))}>Resume</button>}
          {!going &&
            (confirmDelete ? (
              <>
                <button className="danger" onClick={() => client.deleteTrainingJob(job.id).then(onChange)}>Delete</button>
                <button onClick={() => setConfirmDelete(false)}>Cancel</button>
              </>
            ) : (
              <button onClick={() => setConfirmDelete(true)}>Delete</button>
            ))}
        </div>
      </div>
      {confirmDelete && (
        <div className="muted">Delete this run and its checkpoints? A finished model stays on the Models page.</div>
      )}
      {total > 0 && (
        <div className="progress">
          <div style={{ width: `${Math.min(100, (100 * (current.step ?? 0)) / total)}%` }} />
        </div>
      )}
      <div className="muted">
        Step {current.step ?? 0} of {total || "?"} · epoch {(current.epoch ?? 0).toFixed(2)}
        {current.loss != null && ` · loss ${current.loss.toFixed(3)}`}
        {current.eval_loss != null && ` · held-out loss ${current.eval_loss.toFixed(3)}`}
        {job.train_examples != null && ` · ${job.train_examples} training / ${job.eval_examples} held-out examples`}
        {job.dropped > 0 && ` · ${job.dropped} too long, left out`}
        {job.state === "running" && job.eta_seconds != null && ` · about ${duration(job.eta_seconds)} left`}
        {job.state === "done" && ` · the model "${s.name}" is ready in Chat`}
        {job.resumable && job.checkpoint_step != null && ` · can resume from step ${job.checkpoint_step}`}
      </div>
      {job.error && <div className="error">{job.error}</div>}
      {job.state === "error" && log && <pre className="muted">{log}</pre>}
      {points.length > 1 && <LossChart points={points} total={total} />}
    </div>
  );
}

function duration(seconds: number): string {
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 90 * 60) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

const W = 640;
const H = 180;
const PAD = { left: 44, right: 12, top: 10, bottom: 24 };

/** Training loss per step (line) and held-out loss (dots): both should go down together. */
function LossChart({ points, total }: { points: LossPoint[]; total: number }) {
  const values = points.flatMap((p) => [p.loss, p.eval_loss]).filter((v): v is number => v != null);
  const top = Math.max(...values);
  const bottom = Math.min(0, ...values);
  const steps = Math.max(total, points[points.length - 1].step, 1);
  const x = (step: number) => PAD.left + ((W - PAD.left - PAD.right) * step) / steps;
  const y = (v: number) => PAD.top + ((H - PAD.top - PAD.bottom) * (top - v)) / (top - bottom || 1);
  const train = points.filter((p) => p.loss != null).map((p) => `${x(p.step).toFixed(1)},${y(p.loss!).toFixed(1)}`);
  const held = points.filter((p) => p.eval_loss != null);
  const ticks = [bottom, (bottom + top) / 2, top];
  return (
    <svg className="loss-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Loss per training step">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
          <text x={PAD.left - 6} y={y(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
        </g>
      ))}
      <text x={PAD.left} y={H - 6}>0</text>
      <text x={W - PAD.right} y={H - 6} textAnchor="end">step {steps}</text>
      <polyline points={train.join(" ")} className="train" />
      {held.length > 1 && (
        <polyline points={held.map((p) => `${x(p.step).toFixed(1)},${y(p.eval_loss!).toFixed(1)}`).join(" ")} className="held" />
      )}
      {held.map((p) => <circle key={p.step} cx={x(p.step)} cy={y(p.eval_loss!)} r={3} className="held" />)}
      <text x={W - PAD.right} y={PAD.top + 10} textAnchor="end" className="legend">
        <tspan className="train">— training loss</tspan>
        {held.length > 0 && <tspan className="held">{"  ● held-out loss"}</tspan>}
      </text>
    </svg>
  );
}
