import { useCallback, useEffect, useState } from "react";
import {
  CatalogEntry,
  Download,
  EngineClient,
  ExportJob,
  InstalledModel,
  Quantization,
  TrainedModel,
} from "./api";
import { formatSize } from "./format";
import type { ModelFileProgress } from "./types";
import { HubSearch } from "./HubSearch";
import { ModelSettingsForm } from "./ModelSettingsForm";
import { Teachers } from "./Teachers";
import { useTabActive } from "./tabActive";

const FORMATS: Record<InstalledModel["format"], string> = {
  gguf: "GGUF, llama.cpp",
  hf: "PyTorch",
  adapter: "LoRA adapter",
};

export function Models({ client }: { client: EngineClient }) {
  const [models, setModels] = useState<InstalledModel[]>([]);
  const [downloads, setDownloads] = useState<Download[]>([]);
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [trained, setTrained] = useState<TrainedModel[]>([]);
  const [hfToken, setHfToken] = useState(true);
  const [exports, setExports] = useState<ExportJob[]>([]);
  // The fine-tuned model whose export form is open.
  const [exporting, setExporting] = useState<string | null>(null);
  // The model whose settings are open.
  const [configuring, setConfiguring] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Saving a model to a file or uploading one, and what came of it.
  const [transfer, setTransfer] = useState<ModelFileProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => window.tinylm.onModelProgress(setTransfer), []);

  const refresh = useCallback(
    () =>
      Promise.all([client.catalogInfo(), client.trainedModels(), client.exports()]).then(([c, t, x]) => {
        setModels(c.models);
        setDownloads(c.downloads);
        setCatalog(c.catalog);
        setExports(x);
        setHfToken(c.hf_token);
        setTrained(t);
      }, (e) => setError(String(e))),
    [client],
  );

  async function remove(id: string) {
    setConfirmDelete(null);
    setConfiguring(null);
    await client.deleteModel(id).catch((e) => setError(e instanceof Error ? e.message : String(e)));
    await refresh();
  }

  const active = useTabActive();
  useEffect(() => {
    if (!active) return;
    refresh();
    // Poll while downloads are running.
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [refresh, active]);

  // Saves the model to a file on this computer, e.g. one fine-tuned on a RunPod pod.
  async function saveToFile(id: string) {
    setError(null);
    setNote(null);
    setBusy(true);
    setTransfer({ model: id, action: "save", done: 0, total: 0 });
    try {
      const path = await window.tinylm.saveModel(id);
      if (path) setNote(`Saved ${id} to ${path}. Import model with it on another engine.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setTransfer(null);
    }
  }

  // Uploads a saved model or a .gguf file from this computer; the engine then installs it.
  async function importModel() {
    setError(null);
    setNote(null);
    setBusy(true);
    try {
      let upload = await window.tinylm.uploadModel();
      setTransfer(null);
      if (!upload) return;
      setNote(`Installing ${upload.name}…`);
      while (upload.state === "installing" || upload.state === "receiving") {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        upload = await client.modelUpload(upload.id);
      }
      if (upload.state === "error") {
        setNote(null);
        setError(`${upload.name}: ${upload.error}`);
      } else {
        setNote(
          `Installed ${upload.model}.` +
            (upload.missing_base
              ? ` It is fine-tuned from ${upload.missing_base}, which this engine does not have: download or import it too.`
              : ""),
        );
      }
      await refresh();
    } catch (e) {
      setNote(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setTransfer(null);
    }
  }

  async function download(id: string) {
    setError(null);
    try {
      await client.download(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const byId = new Map(models.map((m) => [m.id, m]));
  const trainedIds = new Set(trained.map((m) => m.id));
  // Fine-tuned models and their exports have a table of their own.
  const base = models.filter((m) => !trainedIds.has(m.id));
  const catalogIds = new Set(catalog.map((m) => m.id));
  const hubDownloads = downloads.filter((d) => !catalogIds.has(d.model_id) && d.state !== "done");

  const deleteButtons = (id: string, usedBy: string[] = []) =>
    confirmDelete === id ? (
      <>
        {usedBy.length > 0 && (
          <div className="error">Fine-tuned models {usedBy.join(", ")} need it and stop working.</div>
        )}
        <button className="danger" onClick={() => remove(id)}>Delete</button>{" "}
        <button onClick={() => setConfirmDelete(null)}>Cancel</button>
      </>
    ) : (
      <button onClick={() => setConfirmDelete(id)}>Delete</button>
    );

  const saveButton = (id: string) => (
    <button
      onClick={() => saveToFile(id)}
      disabled={busy}
      title="Saves the model to a file on this computer, for Import model on another engine"
    >
      Save to file
    </button>
  );

  const settingsButton = (id: string) =>
    byId.has(id) && (
      <button onClick={() => setConfiguring(configuring === id ? null : id)}>Settings</button>
    );

  const settingsForm = (id: string) => {
    const model = byId.get(id);
    return (
      configuring === id &&
      model && (
        <ModelSettingsForm
          key={id}
          client={client}
          model={model}
          onDone={() => {
            setConfiguring(null);
            refresh();
          }}
        />
      )
    );
  };

  return (
    <div className="models">
      {error && <p className="error">{error}</p>}
      <div className="heading-row">
        <h3>Installed models</h3>
        <button
          onClick={importModel}
          disabled={busy}
          title="Uploads a model saved with Save to file, or a .gguf file, from this computer to the engine"
        >
          Import model
        </button>
      </div>
      {transfer && <TransferProgress progress={transfer} />}
      {note && <p className="muted">{note}</p>}
      {base.length === 0 ? (
        <p className="muted">No models yet: download one below.</p>
      ) : (
        <table className="installed-models">
          <thead>
            <tr><th>Model</th><th>Format</th><th>Size</th><th /></tr>
          </thead>
          <tbody>
            {base.map((m) => (
              <tr key={m.id}>
                <td>
                  {m.id} {m.loaded && <span className="badge done">loaded</span>}
                  <div className="muted">
                    {m.source
                      ? `${m.source.repo}${m.source.file ? ` · ${m.source.file}` : ""}`
                      : "copied into the models folder"}
                  </div>
                  {Object.keys(m.settings).length > 0 && <div className="muted">{describe(m.settings)}</div>}
                  {settingsForm(m.id)}
                </td>
                <td className="nowrap">{FORMATS[m.format]}</td>
                <td className="nowrap">{formatSize(m.size)}</td>
                <td className="nowrap">
                  {confirmDelete !== m.id && (
                    <>
                      {saveButton(m.id)} {settingsButton(m.id)}
                    </>
                  )}{" "}
                  {deleteButtons(m.id, m.used_by)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
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
                    {m.id} {byId.get(m.id)?.loaded && <span className="badge done">loaded</span>}
                    {m.format === "gguf" ? (
                      <div className="muted">
                        GGUF {m.quantization} for llama.cpp, exported from {m.exported_from} ({m.base}, dataset{" "}
                        {m.dataset})
                      </div>
                    ) : (
                      <div className="muted">LoRA adapter on {m.base}, from dataset {m.dataset}</div>
                    )}
                    {byId.get(m.id)?.settings.collection && (
                      <div className="muted">Answers from the collection {byId.get(m.id)?.settings.collection}</div>
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
                    {settingsForm(m.id)}
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
                    {confirmDelete !== m.id && (
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
                        {saveButton(m.id)} {settingsButton(m.id)}{" "}
                      </>
                    )}
                    {deleteButtons(m.id)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h3>Find models on Hugging Face</h3>
      {hubDownloads.map((d) => (
        <DownloadStatus key={d.model_id} download={d} />
      ))}
      <HubSearch client={client} hfToken={hfToken} downloads={downloads} onDownload={refresh} />

      <h3>Recommended models</h3>
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
                {m.download?.state === "downloading" && <DownloadProgress download={m.download} />}
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

function describe(settings: InstalledModel["settings"]): string {
  return Object.entries(settings)
    .map(([k, v]) => `${k.replace(/_/g, " ")} ${v === true ? "on" : v}`)
    .join(" · ");
}

function TransferProgress({ progress }: { progress: ModelFileProgress }) {
  const share = progress.total ? Math.min(1, progress.done / progress.total) : 0;
  return (
    <div className="download-progress">
      <span>
        {progress.action === "save" ? "Saving" : "Uploading"} {progress.model}
      </span>
      <div className="progress">
        <div style={{ width: `${100 * share}%` }} />
      </div>
      <span className="muted">
        {progress.total ? `${formatSize(progress.done)} of ${formatSize(progress.total)}` : "Starting…"}
      </span>
    </div>
  );
}

function DownloadStatus({ download }: { download: Download }) {
  return (
    <div className="download-status">
      <div>
        {download.state === "error" ? "Download failed" : "Downloading"} {download.model_id}{" "}
        <span className="muted">from {download.repo}</span>
      </div>
      {download.state === "error" ? (
        <div className="error">{download.error}</div>
      ) : (
        <DownloadProgress download={download} />
      )}
    </div>
  );
}

function DownloadProgress({ download }: { download: Download }) {
  const { total, downloaded } = download;
  if (!total) return <div className="muted">{downloaded ? formatSize(downloaded) : "Starting…"}</div>;
  const share = Math.min(1, (downloaded ?? 0) / total);
  return (
    <div className="download-progress">
      <div className="progress">
        <div style={{ width: `${100 * share}%` }} />
      </div>
      <span className="muted">
        {formatSize(downloaded ?? 0)} of {formatSize(total)}
      </span>
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
  const [collection, setCollection] = useState("");
  const [collections, setCollections] = useState<string[]>([]);
  useEffect(() => {
    client.collections().then((c) => setCollections(c.map((x) => x.name)), () => setCollections([]));
  }, [client]);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setError(null);
    try {
      await client.exportModel(model, quantization, name.trim(), collection);
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
      <select
        value={collection}
        onChange={(e) => setCollection(e.target.value)}
        title="Optional: the model folder takes the collection along and answers from it, also when copied to another machine"
      >
        <option value="">Without data</option>
        {collections.map((c) => (
          <option key={c} value={c}>With data: {c}</option>
        ))}
      </select>
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
