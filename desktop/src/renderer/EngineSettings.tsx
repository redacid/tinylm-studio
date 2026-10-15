import { useCallback, useEffect, useState } from "react";
import { Check, Field } from "./Settings";
import type {
  EngineConnection,
  EngineConnectionChanges,
  EngineMode,
  EngineState,
  Pod,
} from "./types";

const MODES: { mode: EngineMode; title: string; text: string }[] = [
  {
    mode: "local",
    title: "On this computer",
    text: "The app starts the engine itself, with this computer's CPU and GPU.",
  },
  {
    mode: "remote",
    title: "Remote engine",
    text: "An engine started elsewhere (a server, a cluster) with TINYLM_API_KEY.",
  },
  {
    mode: "runpod",
    title: "RunPod",
    text: "A GPU pod on runpod.io running the engine image; the app starts and stops it.",
  },
];

// RunPod's GPU type ids; any other id from runpod.io works too.
const GPUS = [
  "NVIDIA GeForce RTX 4090",
  "NVIDIA GeForce RTX 3090",
  "NVIDIA RTX A4000",
  "NVIDIA RTX A5000",
  "NVIDIA RTX A6000",
  "NVIDIA L4",
  "NVIDIA A40",
  "NVIDIA L40S",
  "NVIDIA A100 80GB PCIe",
  "NVIDIA H100 80GB HBM3",
];

const ACTIVE: Pod["status"][] = ["PROVISIONING", "STARTING", "RUNNING"];

/** ipcRenderer.invoke wraps errors as "Error invoking remote method 'x': Error: message". */
function message(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error);
  return text.replace(
    /^Error invoking remote method '[^']+': (\w*Error: )?/,
    "",
  );
}

/** Where the engine runs: on this computer, at an address, or in a RunPod pod. */
export function EngineSettingsForm({ engine }: { engine: EngineState }) {
  const [saved, setSaved] = useState<EngineConnection | null>(null);
  const [mode, setMode] = useState<EngineMode>("local");
  const [url, setUrl] = useState("");
  // Keys are never shown: typing one replaces it, an empty field keeps it.
  const [remoteKey, setRemoteKey] = useState("");
  const [runpodKey, setRunpodKey] = useState("");
  const [engineKey, setEngineKey] = useState("");
  const [runpod, setRunpod] = useState<EngineConnection["runpod"] | null>(null);
  const [name, setName] = useState("tinylm-engine");
  const [pod, setPod] = useState<Pod | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmTerminate, setConfirmTerminate] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loaded = (c: EngineConnection) => {
    setSaved(c);
    setMode(c.mode);
    setUrl(c.remote.url);
    setRunpod(c.runpod);
    setRemoteKey("");
    setRunpodKey("");
    setEngineKey("");
  };
  useEffect(() => {
    window.tinylm.getConnection().then(loaded);
  }, []);

  const podId = saved?.runpod.podId ?? "";
  const refreshPod = useCallback(async () => {
    if (!podId || !saved?.runpod.apiKeySet) {
      setPod(null);
      return;
    }
    try {
      setPod(await window.tinylm.runpod("pod"));
    } catch (e) {
      setError(message(e));
    }
  }, [podId, saved?.runpod.apiKeySet]);
  useEffect(() => {
    refreshPod();
    if (!podId) return;
    const timer = setInterval(refreshPod, 10_000);
    return () => clearInterval(timer);
  }, [podId, refreshPod]);

  if (!saved || !runpod) return <p className="muted">Loading…</p>;

  const changes = (): EngineConnectionChanges => ({
    mode,
    remote: { url, ...(remoteKey ? { apiKey: remoteKey } : {}) },
    runpod: {
      ...runpod,
      ...(runpodKey ? { apiKey: runpodKey } : {}),
      ...(engineKey ? { engineKey } : {}),
    },
  });
  const setPodField = (change: Partial<EngineConnection["runpod"]>) =>
    setRunpod({ ...runpod, ...change });

  async function act(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  const save = (reconnect: boolean) =>
    act(async () =>
      loaded(await window.tinylm.setConnection(changes(), reconnect)),
    );
  // Pod actions use what is in the form, so it is saved first.
  const podAction = (action: "start" | "stop" | "terminate") =>
    act(async () => {
      loaded(await window.tinylm.setConnection(changes(), false));
      await window.tinylm.runpod(action);
      setConfirmTerminate(false);
      if (action === "terminate") loaded(await window.tinylm.getConnection());
      await refreshPod();
    });
  const create = () =>
    act(async () => {
      loaded(await window.tinylm.setConnection(changes(), false));
      setPod(await window.tinylm.runpod("create", name.trim()));
      loaded(await window.tinylm.getConnection());
    });

  const connected = engine.phase === "ready" && saved.mode === mode;
  const keyPlaceholder = (set: boolean) => (set ? "set: type to replace" : "");

  return (
    <div className="settings-form">
      <h3>Engine</h3>
      <p className="muted">
        Where models, data, training and exports live. Each engine has its own:
        switching shows the other engine's models and collections, nothing is
        copied.
      </p>

      <h4>Where it runs</h4>
      <div className="provider-list engine-modes">
        {MODES.map((m) => (
          <label
            key={m.mode}
            className={`provider ${mode === m.mode ? "active" : ""}`}
          >
            <input
              type="radio"
              name="engine-mode"
              checked={mode === m.mode}
              onChange={() => setMode(m.mode)}
            />
            <span>
              <strong>{m.title}</strong>
              <small className="muted">{m.text}</small>
            </span>
          </label>
        ))}
      </div>

      {mode === "remote" && (
        <>
          <Field
            label="Engine address"
            hint="Where the engine's API answers, e.g. https://engine.example.com"
          >
            <input
              value={url}
              placeholder="https://"
              onChange={(e) => setUrl(e.target.value)}
            />
          </Field>
          <Field
            label="API key"
            hint="The TINYLM_API_KEY the engine was started with."
          >
            <input
              type="password"
              value={remoteKey}
              placeholder={keyPlaceholder(saved.remote.apiKeySet)}
              onChange={(e) => setRemoteKey(e.target.value)}
            />
          </Field>
        </>
      )}

      {mode === "runpod" && (
        <>
          <Field
            label="RunPod API key"
            hint="runpod.io console › Credentials › API Keys, with All permissions. Kept only on this computer."
          >
            <input
              type="password"
              value={runpodKey}
              placeholder={keyPlaceholder(saved.runpod.apiKeySet)}
              onChange={(e) => setRunpodKey(e.target.value)}
            />
          </Field>

          <h4>Pod</h4>
          {podId ? (
            <div className="pod">
              <div className="pod-head">
                <strong>{pod?.name || podId}</strong>
                <span
                  className={`badge pod-${(pod?.status ?? "unknown").toLowerCase()}`}
                >
                  {pod?.status.toLowerCase() ?? "…"}
                </span>
                <span className="muted">{podId}</span>
              </div>
              {pod && (
                <div className="muted">
                  {[
                    pod.gpu,
                    pod.cost !== null &&
                      `$${pod.cost.toFixed(2)}/hour while running`,
                    pod.uptime !== null &&
                      `up ${Math.round(pod.uptime / 60)} min`,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                  <div>{pod.url}</div>
                </div>
              )}
              <div className="buttons-row">
                {pod?.status === "EXITED" && (
                  <button
                    className="primary"
                    disabled={busy}
                    onClick={() => podAction("start")}
                  >
                    Start
                  </button>
                )}
                {pod && ACTIVE.includes(pod.status) && (
                  <button disabled={busy} onClick={() => podAction("stop")}>
                    Stop
                  </button>
                )}
                {confirmTerminate ? (
                  <span className="danger">
                    Delete the pod and its pod volume? A network volume stays.{" "}
                    <button
                      className="danger"
                      disabled={busy}
                      onClick={() => podAction("terminate")}
                    >
                      Terminate
                    </button>{" "}
                    <button onClick={() => setConfirmTerminate(false)}>
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    disabled={busy}
                    onClick={() => setConfirmTerminate(true)}
                  >
                    Terminate
                  </button>
                )}
              </div>
            </div>
          ) : (
            <>
              <p className="muted">
                No pod yet. Create one from the engine image (see the user guide
                for building and pushing it), or enter the id of a pod you made
                on runpod.io below.
              </p>
              <div className="settings-grid">
                <Field label="Name" hint="Shown on runpod.io.">
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </Field>
                <Field
                  label="Engine image"
                  hint="The CUDA image you pushed, e.g. docker.io/you/tinylm-engine:cuda."
                >
                  <input
                    value={runpod.image}
                    onChange={(e) => setPodField({ image: e.target.value })}
                  />
                </Field>
                <Field
                  label="GPU"
                  hint="A RunPod GPU type id; the list has common ones, any from runpod.io works."
                >
                  <input
                    list="runpod-gpus"
                    value={runpod.gpu}
                    onChange={(e) => setPodField({ gpu: e.target.value })}
                  />
                </Field>
                <Field
                  label="Cloud"
                  hint="Secure: RunPod's data centers. Community: cheaper, hosted by others."
                >
                  <select
                    value={runpod.cloud}
                    onChange={(e) =>
                      setPodField({
                        cloud: e.target.value as "SECURE" | "COMMUNITY",
                      })
                    }
                  >
                    <option value="SECURE">Secure</option>
                    <option value="COMMUNITY">Community</option>
                  </select>
                </Field>
                <Field
                  label="Network volume"
                  hint="Optional id of a network volume: the data outlives the pod and can move to a new one."
                >
                  <input
                    value={runpod.networkVolume}
                    onChange={(e) =>
                      setPodField({ networkVolume: e.target.value })
                    }
                  />
                </Field>
                <Field
                  label="Pod volume, GB"
                  hint="Without a network volume: models and data, kept while the pod exists."
                >
                  <input
                    type="number"
                    min={10}
                    max={2000}
                    disabled={!!runpod.networkVolume}
                    value={runpod.volumeSize}
                    onChange={(e) =>
                      setPodField({ volumeSize: Number(e.target.value) })
                    }
                  />
                </Field>
                <Field
                  label="Container disk, GB"
                  hint="The image and temporary files; emptied when the pod stops."
                >
                  <input
                    type="number"
                    min={20}
                    max={500}
                    value={runpod.disk}
                    onChange={(e) =>
                      setPodField({ disk: Number(e.target.value) })
                    }
                  />
                </Field>
              </div>
              <datalist id="runpod-gpus">
                {GPUS.map((g) => (
                  <option key={g} value={g} />
                ))}
              </datalist>
              <div className="buttons-row">
                <button
                  className="primary"
                  disabled={
                    busy ||
                    !runpod.image.trim() ||
                    !(runpodKey || saved.runpod.apiKeySet)
                  }
                  onClick={create}
                >
                  Create pod
                </button>
              </div>
            </>
          )}
          <details className="pod-existing">
            <summary>Use a pod made on runpod.io</summary>
            <div className="settings-grid">
              <Field
                label="Pod id"
                hint="From runpod.io › Pods; it must expose port 8765 over HTTP."
              >
                <input
                  value={runpod.podId}
                  onChange={(e) =>
                    setPodField({ podId: e.target.value.trim() })
                  }
                />
              </Field>
              <Field
                label="Engine API key"
                hint="The pod's TINYLM_API_KEY. Set by the app for pods it creates."
              >
                <input
                  type="password"
                  value={engineKey}
                  placeholder={keyPlaceholder(saved.runpod.engineKeySet)}
                  onChange={(e) => setEngineKey(e.target.value)}
                />
              </Field>
            </div>
          </details>
          <Check
            checked={runpod.stopOnQuit}
            onChange={(v) => setPodField({ stopOnQuit: v })}
          >
            Stop the pod when the app closes (a running pod is billed by the
            second; its volume is kept)
          </Check>
        </>
      )}

      {error && <p className="error">{error}</p>}
      {engine.phase === "error" && saved.mode === mode && (
        <p className="error">{engine.message}</p>
      )}
      <div className="buttons-row">
        <button className="primary" disabled={busy} onClick={() => save(true)}>
          {connected ? "Save and reconnect" : "Connect"}
        </button>
        {mode !== "local" && (
          <button disabled={busy} onClick={() => save(false)}>
            Save
          </button>
        )}
      </div>
    </div>
  );
}
