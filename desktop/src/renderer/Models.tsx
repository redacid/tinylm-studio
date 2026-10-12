import { useCallback, useEffect, useState } from "react";
import { CatalogEntry, EngineClient, TrainedModel } from "./api";
import { Teachers } from "./Teachers";

export function Models({ client }: { client: EngineClient }) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [trained, setTrained] = useState<TrainedModel[]>([]);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    () =>
      Promise.all([client.catalog(), client.trainedModels()]).then(([c, t]) => {
        setCatalog(c);
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
                    <div className="muted">LoRA adapter on {m.base}, from dataset {m.dataset}</div>
                    {!m.base_installed && <div className="error">The base model {m.base} is not installed.</div>}
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
                      <button onClick={() => setConfirmDelete(m.id)}>Delete</button>
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
                {m.gated && <div className="muted">Requires a Hugging Face token (TINYLM_HF_TOKEN).</div>}
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
