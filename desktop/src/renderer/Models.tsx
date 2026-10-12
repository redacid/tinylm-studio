import { useCallback, useEffect, useState } from "react";
import { CatalogEntry, EngineClient, ExportJob, Quantization, TrainedModel } from "./api";
import { Teachers } from "./Teachers";

export function Models({ client }: { client: EngineClient }) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [trained, setTrained] = useState<TrainedModel[]>([]);
  const [hfToken, setHfToken] = useState(true);
  const [exports, setExports] = useState<ExportJob[]>([]);
  // The fine-tuned model whose export form is open.
  const [exporting, setExporting] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    () =>
      Promise.all([client.catalogInfo(), client.trainedModels(), client.exports()]).then(([c, t, x]) => {
        setCatalog(c.catalog);
        setExports(x);
        setHfToken(c.hf_token);
        setTrained(t);
      }, (e) => setError(String(e))),
    [client],
  );

  async function remove(id: string) {
    setConfirmDelete(null);
    await client.deleteModel(id).catch((e) => setError(e instanceof Error ? e.message : String(e)));
    await refresh();
  }

  useEffect(() => {
    refresh();
    // Poll while downloads are running.
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [refresh]);

  async function download(id: string) {
    setError(null);
    try {
      await client.download(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="models">
      {error && <p className="error">{error}</p>}
      {trained.length > 0 && (
        <>
          <h3>Fine-tuned models</h3>
          <table>
            <thead>
              <tr><th>Model</th><th>Trained</th><th>Loss</th><th /></tr>
            </thead>
            <tbody>
              {trained.map((m) => (
                <tr key={m.id}>
                  <td>
                    {m.id}
                    {m.format === "gguf" ? (
                      <div className="muted">
                        GGUF {m.quantization} for llama.cpp, exported from {m.exported_from} ({m.base}, dataset{" "}
                        {m.dataset})
                      </div>
                    ) : (
                      <div className="muted">LoRA adapter on {m.base}, from dataset {m.dataset}</div>
                    )}
                    {!m.base_installed && <div className="error">The base model {m.base} is not installed.</div>}
                    <ExportStatus job={[...exports].reverse().find((x) => x.model === m.id)} />
                    {exporting === m.id && (
                      <ExportForm
                        client={client}
                        model={m.id}
                        onDone={() => {
                          setExporting(null);
                          refresh();
                        }}
                      />
                    )}
                  </td>
                  <td className="nowrap">
                    {new Date(m.created_at).toLocaleString()}
                    <div className="muted">
                      {m.settings.epochs} epochs · lr {m.settings.learning_rate} · rank {m.settings.lora_rank}
                    </div>
                  </td>
                  <td className="nowrap">
                    {m.loss != null && `training ${m.loss.toFixed(3)}`}
                    {m.eval_loss != null && <div className="muted">held-out {m.eval_loss.toFixed(3)}</div>}
                  </td>
                  <td className="nowrap">
                    {confirmDelete === m.id ? (
                      <>
                        <button className="danger" onClick={() => remove(m.id)}>Delete</button>{" "}
                        <button onClick={() => setConfirmDelete(null)}>Cancel</button>
                      </>
                    ) : (
                      <>
                        {m.format === "adapter" && (
                          <button
                            title="Merge into the base model and save as one GGUF file: much faster answers through llama.cpp"
                            disabled={!m.base_installed || exports.some((x) => x.state === "running")}
                            onClick={() => setExporting(exporting === m.id ? null : m.id)}
                          >
                            Export to GGUF
                          </button>
                        )}{" "}
                        <button onClick={() => setConfirmDelete(m.id)}>Delete</button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Base models</h3>
        </>
      )}
      <table>
        <thead>
          <tr><th>Model</th><th>Size</th><th>Description</th><th /></tr>
        </thead>
        <tbody>
          {catalog.map((m) => (
            <tr key={m.id}>
              <td>{m.id}<div className="muted">{m.repo}</div></td>
              <td>{m.params}</td>
              <td>
                {m.description}
                {m.gated && !m.installed && (
                  <div className={hfToken ? "muted" : "error"}>
                    {hfToken
                      ? "Gated: accept the license on Hugging Face first."
                      : "Gated: needs a Hugging Face token (HF_TOKEN in .env) and an accepted license."}{" "}
                    <a href={`https://huggingface.co/${m.repo}`} target="_blank" rel="noreferrer">
                      Model page
                    </a>
                  </div>
                )}
                {m.download?.state === "error" && <div className="error">{m.download.error}</div>}
              </td>
              <td>
                {m.installed ? (
                  "Installed"
                ) : m.download?.state === "downloading" ? (
                  "Downloading…"
                ) : (
                  <button onClick={() => download(m.id)}>Download</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">
        Any GGUF or Hugging Face-format model folder copied into the models directory also appears here.
      </p>
      <Teachers client={client} />
    </div>
  );
}

const QUANTIZATIONS: { value: Quantization; label: string }[] = [
  { value: "Q8_0", label: "Q8_0: 8-bit, practically no loss (recommended)" },
  { value: "Q4_K_M", label: "Q4_K_M: 4-bit, half the size and faster, small loss" },
  { value: "F16", label: "F16: 16-bit, no loss, largest" },
];

function ExportForm({ client, model, onDone }: { client: EngineClient; model: string; onDone: () => void }) {
  const [quantization, setQuantization] = useState<Quantization>("Q8_0");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setError(null);
    try {
      await client.exportModel(model, quantization, name.trim());
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="export-form">
      <select value={quantization} onChange={(e) => setQuantization(e.target.value as Quantization)}>
        {QUANTIZATIONS.map((q) => (
          <option key={q.value} value={q.value}>{q.label}</option>
        ))}
      </select>
      <input placeholder={`${model}-${quantization.toLowerCase()}`} value={name} onChange={(e) => setName(e.target.value)} />
      <button onClick={start}>Start export</button>
      {error && <div className="error">{error}</div>}
    </div>
  );
}

const STEP_LABELS: Record<string, string> = {
  tools: "Getting the converter",
  merge: "Merging the adapter into the base model",
  convert: "Converting to GGUF",
  quantize: "Quantizing",
  save: "Saving",
};

function ExportStatus({ job }: { job?: ExportJob }) {
  if (!job) return null;
  if (job.state === "running") {
    const index = job.step ? job.steps.indexOf(job.step) : 0;
    return (
      <div className="export-status">
        <div className="muted">
          Exporting {job.name}: {STEP_LABELS[job.step ?? "tools"]} ({index + 1} of {job.steps.length})…
        </div>
        <div className="progress">
          <div style={{ width: `${(100 * index) / job.steps.length}%` }} />
        </div>
        {job.notes.map((note, i) => (
          <div key={i} className="muted">{note}</div>
        ))}
      </div>
    );
  }
  if (job.state === "done") {
    const megabytes = job.size ? ` (${formatSize(job.size)})` : "";
    return (
      <div className="muted">
        Exported as {job.name}
        {megabytes}.{job.notes.filter((n) => n.includes("pre-tokenizer")).map((n) => ` ${n}`)}
      </div>
    );
  }
  return <div className="error">Export {job.state === "interrupted" ? "interrupted" : "failed"}: {job.error}</div>;
}

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}
