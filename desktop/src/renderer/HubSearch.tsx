import { FormEvent, useState } from "react";
import { Download, EngineClient, HubFiles, HubFormat, HubModel } from "./api";
import { formatSize } from "./format";
import { Icon } from "./Icon";

/** Finds open models on Hugging Face and downloads one GGUF file or the PyTorch weights. */
export function HubSearch({
  client,
  hfToken,
  downloads,
  onDownload,
}: {
  client: EngineClient;
  hfToken: boolean;
  /** Downloads of the engine, to show which files are already coming. */
  downloads: Download[];
  onDownload: () => void;
}) {
  const [query, setQuery] = useState("");
  const [format, setFormat] = useState<HubFormat>("gguf");
  const [results, setResults] = useState<HubModel[] | null>(null);
  const [searching, setSearching] = useState(false);
  // The repo whose files are shown, and its files once read.
  const [open, setOpen] = useState<string | null>(null);
  const [files, setFiles] = useState<HubFiles | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function search(e?: FormEvent) {
    e?.preventDefault();
    setError(null);
    setSearching(true);
    setOpen(null);
    try {
      setResults(await client.searchHub(query, format));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  }

  async function showFiles(repo: string) {
    if (open === repo) {
      setOpen(null);
      return;
    }
    setOpen(repo);
    setFiles(null);
    setError(null);
    try {
      setFiles(await client.hubFiles(repo));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function download(repo: string, file?: string) {
    setError(null);
    try {
      await client.downloadFromHub(repo, file);
      onDownload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const coming = (repo: string) =>
    downloads.some((d) => d.repo === repo && d.state === "downloading")
      ? "Downloading…"
      : null;

  return (
    <div className="hub-search">
      <form className="buttons-row" onSubmit={search}>
        <input
          placeholder="Search Hugging Face, e.g. gemma 3 4b, qwen3, llama 3.2 1b"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select
          value={format}
          onChange={(e) => setFormat(e.target.value as HubFormat)}
        >
          <option value="gguf">GGUF: fast answers through llama.cpp</option>
          <option value="hf">PyTorch: can be fine-tuned</option>
        </select>
        <button type="submit" disabled={searching}>
          <Icon name={searching ? "loader-2" : "search"} spin={searching} />
          {searching ? "Searching…" : "Search"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {results && results.length === 0 && (
        <p className="muted">Nothing found.</p>
      )}
      {results && results.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Size</th>
              <th>Downloads</th>
              <th>Updated</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {results.map((m) => (
              <tr key={m.repo}>
                <td>
                  <a
                    href={`https://huggingface.co/${m.repo}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {m.repo}
                  </a>
                  {m.gated && (
                    <div className={hfToken ? "muted" : "error"}>
                      Gated: accept the license on Hugging Face first
                      {hfToken ? "." : ", and set HF_TOKEN in .env."}
                    </div>
                  )}
                  {open === m.repo && (
                    <RepoFiles
                      files={files}
                      format={format}
                      busy={coming(m.repo)}
                      onDownload={(file) => download(m.repo, file)}
                    />
                  )}
                </td>
                <td className="nowrap">
                  {m.params ? formatParams(m.params) : ""}
                  {m.context_length && (
                    <div className="muted">
                      context {m.context_length.toLocaleString()}
                    </div>
                  )}
                </td>
                <td className="nowrap">
                  {m.downloads.toLocaleString()}
                  <div className="muted">♥ {m.likes.toLocaleString()}</div>
                </td>
                <td className="nowrap">
                  {m.updated ? new Date(m.updated).toLocaleDateString() : ""}
                </td>
                <td>
                  <button onClick={() => showFiles(m.repo)}>
                    <Icon name={open === m.repo ? "eye-off" : "list-check"} />
                    {open === m.repo ? "Hide" : "Choose"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function RepoFiles({
  files,
  format,
  busy,
  onDownload,
}: {
  files: HubFiles | null;
  format: HubFormat;
  busy: string | null;
  onDownload: (file?: string) => void;
}) {
  if (!files) return <div className="muted">Reading the files…</div>;
  if (format === "hf") {
    return files.hf ? (
      <div className="repo-files">
        <button disabled={!!busy} onClick={() => onDownload()}>
          <Icon name={busy ? "loader-2" : "download"} spin={!!busy} />
          {busy ?? `Download (${formatSize(files.hf.size)})`}
        </button>
      </div>
    ) : (
      <div className="error">
        No PyTorch (safetensors) weights in this repo.
      </div>
    );
  }
  if (files.gguf.length === 0)
    return <div className="error">No GGUF files in this repo.</div>;
  return (
    <table className="repo-files">
      <tbody>
        {files.gguf.map((f) => (
          <tr key={f.file}>
            <td>{f.quantization ?? "—"}</td>
            <td className="muted">
              {f.file}
              {f.parts > 1 && ` (${f.parts} parts)`}
            </td>
            <td className="nowrap">{formatSize(f.size)}</td>
            <td>
              <button disabled={!!busy} onClick={() => onDownload(f.file)}>
                <Icon name="download" />Download
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  return `${Math.round(n / 1e6)}M`;
}
