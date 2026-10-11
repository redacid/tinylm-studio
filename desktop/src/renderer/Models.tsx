import { useCallback, useEffect, useState } from "react";
import { CatalogEntry, EngineClient } from "./api";

export function Models({ client }: { client: EngineClient }) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    () => client.catalog().then(setCatalog, (e) => setError(String(e))),
    [client],
  );

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
        Any Hugging Face-format model folder copied into the models directory also appears here.
      </p>
    </div>
  );
}
