import { useCallback, useEffect, useState } from "react";
import { Check, Field } from "./Settings";
import type {
  Availability,
  EngineConnection,
  EngineConnectionChanges,
  EngineMode,
  EngineState,
  Pod,
  RunPodCatalog,
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
    text: "A GPU or CPU pod on runpod.io running the engine image; the app starts and stops it.",
  },
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
  const [catalog, setCatalog] = useState<RunPodCatalog | null>(null);
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

  // The lists of GPUs, CPUs, data centers and volumes, once there is a key and no pod yet.
  const cloud = runpod?.cloud;
  const keySet = !!saved?.runpod.apiKeySet;
  useEffect(() => {
    if (mode !== "runpod" || !keySet || podId || !cloud) return;
    let current = true;
    window.tinylm
      .runpod("catalog", cloud)
      .then((c) => current && setCatalog(c))
      .catch((e) => current && setError(message(e)));
    return () => {
      current = false;
    };
  }, [mode, keySet, podId, cloud]);

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

  // The lists depend on the key and the cloud, so the form is saved first.
  const loadCatalog = () =>
    act(async () => {
      loaded(await window.tinylm.setConnection(changes(), false));
      setCatalog(await window.tinylm.runpod("catalog", runpod.cloud));
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
                    pod.hardware,
                    pod.dataCenter,
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
              <PodForm
                name={name}
                setName={setName}
                spec={runpod}
                set={setPodField}
                catalog={catalog}
              />
              <div className="buttons-row">
                <button
                  disabled={busy || !(runpodKey || saved.runpod.apiKeySet)}
                  onClick={loadCatalog}
                >
                  {catalog ? "Refresh lists" : "Load lists from RunPod"}
                </button>
                {catalog && (
                  <span className="muted">
                    Prices and what is free now, from RunPod.
                  </span>
                )}
              </div>
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

type Spec = EngineConnection["runpod"];

const AVAILABILITY: Record<Availability, string> = {
  HIGH: "many free",
  MEDIUM: "some free",
  LOW: "few free",
  NONE: "none free",
};
/** What RunPod's availability levels mean; it gives no exact counts. */
const AVAILABILITY_HINT: Record<Availability, string> = {
  HIGH: "RunPod: HIGH. Plenty free now; a pod should be created at once.",
  MEDIUM: "RunPod: MEDIUM. Some free now, not many.",
  LOW: "RunPod: LOW. Few left; creating a pod can fail if others take them first.",
  NONE: "RunPod: NONE. None free now; a pod with it cannot be created. Try later or another one.",
};

const RANK: Record<Availability, number> = {
  HIGH: 0,
  MEDIUM: 1,
  LOW: 2,
  NONE: 3,
};

const money = (dollars: number) =>
  `$${dollars < 1 ? dollars.toFixed(3).replace(/0$/, "") : dollars.toFixed(2)}`;

// The CUDA version the engine image is built for (see runpod.ts).
const MIN_CUDA = "12.6";

/** "NVIDIA", "AMD"; RunPod says UNKNOWN when it does not know. */
const maker = (manufacturer: string) =>
  manufacturer === "UNKNOWN" ? "Unknown maker" : manufacturer;

/** Powers of two from min to max: the vCPU counts RunPod accepts. */
function vcpuCounts(min: number, max: number): number[] {
  const counts = [];
  for (let n = 2; n <= max; n *= 2) if (n >= min) counts.push(n);
  return counts;
}

/** How a new pod is made: GPU or CPU, where, and on which volume. With the catalog loaded the
 * choices come from RunPod, with prices and what is free now; without it, ids are typed. */
function PodForm({
  name,
  setName,
  spec,
  set,
  catalog,
}: {
  name: string;
  setName: (name: string) => void;
  spec: Spec;
  set: (change: Partial<Spec>) => void;
  catalog: RunPodCatalog | null;
}) {
  const gpu = catalog?.gpus.find((g) => g.id === spec.gpu);
  const cpu = catalog?.cpus.find((c) => c.id === spec.cpu);
  const volume = catalog?.volumes.find((v) => v.id === spec.networkVolume);
  const chosen = spec.kind === "gpu" ? gpu : cpu;
  const centerName = (id: string) =>
    catalog?.dataCenters.find((d) => d.id === id)?.name ?? id;

  // Data centers where the chosen GPU or CPU is free now; a network volume pins its own.
  const centers = chosen
    ? chosen.dataCenters
        .filter((d) => d.availability !== "NONE")
        .sort((a, b) => RANK[a.availability] - RANK[b.availability])
    : [];
  const price =
    spec.kind === "gpu"
      ? gpu?.price
      : cpu
        ? cpu.pricePerVcpu * spec.vcpus
        : undefined;
  // NVIDIA first (the image needs CUDA), then free ones, then the cheapest.
  const sortedGpus = [...(catalog?.gpus ?? [])].sort(
    (a, b) =>
      Number(a.manufacturer !== "NVIDIA") -
        Number(b.manufacturer !== "NVIDIA") ||
      RANK[a.availability] - RANK[b.availability] ||
      a.price - b.price,
  );
  const sortedCpus = [...(catalog?.cpus ?? [])].sort(
    (a, b) =>
      RANK[a.availability] - RANK[b.availability] ||
      a.pricePerVcpu - b.pricePerVcpu,
  );
  const chooseCpu = (id: string) => {
    const flavor = catalog?.cpus.find((c) => c.id === id);
    const counts = flavor ? vcpuCounts(flavor.vcpu.min, flavor.vcpu.max) : [];
    set({
      cpu: id,
      vcpus: counts.includes(spec.vcpus) ? spec.vcpus : (counts[0] ?? 2),
    });
  };
  const chooseVolume = (id: string) => {
    const v = catalog?.volumes.find((x) => x.id === id);
    set({ networkVolume: id, ...(v ? { dataCenter: v.dataCenter } : {}) });
  };

  return (
    <>
      <div className="settings-grid">
        <Field label="Name" hint="Shown on runpod.io.">
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field
          label="Engine image"
          hint={`The ${spec.kind === "gpu" ? "cuda" : "cpu"} image make image-push printed, e.g. docker.io/you/tinylm-engine:0.1.0-5a0d901-${spec.kind === "gpu" ? "cuda" : "cpu"}.`}
        >
          <input
            value={spec.image}
            onChange={(e) => set({ image: e.target.value })}
          />
        </Field>
        <Field
          label="Pod type"
          hint="GPU: answers and fine-tuning on the GPU. CPU: cheaper, for answers with small GGUF models."
        >
          <select
            value={spec.kind}
            onChange={(e) =>
              set({ kind: e.target.value as Spec["kind"], dataCenter: "" })
            }
          >
            <option value="gpu">GPU</option>
            <option value="cpu">CPU only</option>
          </select>
        </Field>
        <Field
          label="Cloud"
          hint={
            spec.kind === "gpu"
              ? "Secure: RunPod's data centers. Community: cheaper, hosted by others."
              : "CPU pods run in RunPod's own data centers."
          }
        >
          <select
            value={spec.kind === "gpu" ? spec.cloud : "SECURE"}
            disabled={spec.kind === "cpu"}
            onChange={(e) =>
              set({
                cloud: e.target.value as Spec["cloud"],
                dataCenter: volume ? spec.dataCenter : "",
              })
            }
          >
            <option value="SECURE">Secure</option>
            <option value="COMMUNITY">Community</option>
          </select>
        </Field>

        {spec.kind === "gpu" ? (
          catalog ? (
            <div className="settings-field wide">
              <span>GPU</span>
              <GpuTable
                gpus={sortedGpus}
                chosen={spec.gpu}
                choose={(id) => set({ gpu: id, dataCenter: "" })}
              />
              <small className="muted">
                {gpu
                  ? `${maker(gpu.manufacturer)} ${gpu.name}, ${gpu.memory} GB of video memory; ${AVAILABILITY[gpu.availability]} now.`
                  : `${spec.gpu || "None"} is not in RunPod's list for this cloud: choose one.`}{" "}
                The green price is the cheapest GPU free now; hover a column or
                value for what it means. Only hosts with CUDA {MIN_CUDA}+, which
                the image needs.
              </small>
            </div>
          ) : (
            <Field
              label="GPU"
              hint="A RunPod GPU type id; load the lists to choose with prices."
            >
              <input
                value={spec.gpu}
                onChange={(e) => set({ gpu: e.target.value })}
              />
            </Field>
          )
        ) : (
          <>
            <Field
              label="CPU"
              hint={
                cpu
                  ? `${cpu.name}: ${AVAILABILITY[cpu.availability]} now.`
                  : catalog
                    ? "Choose a CPU type."
                    : "A RunPod CPU flavor id, e.g. cpu3c; load the lists to choose."
              }
            >
              {catalog ? (
                <select
                  value={spec.cpu}
                  onChange={(e) => chooseCpu(e.target.value)}
                >
                  {!cpu && (
                    <option value={spec.cpu}>{spec.cpu || "Choose"}</option>
                  )}
                  {sortedCpus.map((c) => (
                    <option
                      key={c.id}
                      value={c.id}
                      title={`${c.vcpu.min}–${c.vcpu.max} vCPUs, ${c.ramGbPerVcpu} GB of memory per vCPU. ${AVAILABILITY_HINT[c.availability]}`}
                    >
                      {c.name} ({c.group}) · {money(c.pricePerVcpu)}/vCPU·h ·{" "}
                      {AVAILABILITY[c.availability]}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={spec.cpu}
                  onChange={(e) => set({ cpu: e.target.value.trim() })}
                />
              )}
            </Field>
            <Field
              label="vCPUs"
              hint={
                cpu
                  ? `${spec.vcpus * cpu.ramGbPerVcpu} GB of memory with ${spec.vcpus} vCPUs.`
                  : "A power of two."
              }
            >
              <select
                value={spec.vcpus}
                onChange={(e) => set({ vcpus: Number(e.target.value) })}
              >
                {(cpu
                  ? vcpuCounts(cpu.vcpu.min, cpu.vcpu.max)
                  : [2, 4, 8, 16, 32]
                ).map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </Field>
          </>
        )}

        <Field
          label="Network volume"
          hint={
            volume
              ? `Keeps the data when the pod is gone; the pod runs in ${centerName(volume.dataCenter)}, where the volume is.`
              : "None: the data is on the pod" +
                (spec.kind === "gpu"
                  ? "'s volume, deleted with the pod."
                  : "'s container disk, lost when it stops.")
          }
        >
          {catalog ? (
            <select
              value={spec.networkVolume}
              onChange={(e) => chooseVolume(e.target.value)}
            >
              <option value="">None</option>
              {!volume && spec.networkVolume && (
                <option value={spec.networkVolume}>{spec.networkVolume}</option>
              )}
              {catalog.volumes.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name} · {v.size} GB · {centerName(v.dataCenter)}
                </option>
              ))}
            </select>
          ) : (
            <input
              value={spec.networkVolume}
              placeholder="volume id"
              onChange={(e) => set({ networkVolume: e.target.value.trim() })}
            />
          )}
        </Field>
        <Field
          label="Data center"
          hint={
            volume
              ? "Set by the network volume."
              : catalog
                ? "Where the chosen hardware is free now; Any lets RunPod choose."
                : "A data center id, e.g. EU-RO-1; empty lets RunPod choose."
          }
        >
          {catalog ? (
            <select
              value={spec.dataCenter}
              disabled={!!volume}
              onChange={(e) => set({ dataCenter: e.target.value })}
            >
              <option value="">Any</option>
              {spec.dataCenter &&
                !centers.some((d) => d.id === spec.dataCenter) && (
                  <option value={spec.dataCenter}>
                    {centerName(spec.dataCenter)} (none free now)
                  </option>
                )}
              {centers.map((d) => (
                <option
                  key={d.id}
                  value={d.id}
                  title={AVAILABILITY_HINT[d.availability]}
                >
                  {centerName(d.id)} · {AVAILABILITY[d.availability]}
                </option>
              ))}
            </select>
          ) : (
            <input
              value={spec.dataCenter}
              onChange={(e) => set({ dataCenter: e.target.value.trim() })}
            />
          )}
        </Field>
        {spec.kind === "gpu" && (
          <Field
            label="Pod volume, GB"
            hint="Without a network volume: models and data, kept while the pod exists."
          >
            <input
              type="number"
              min={10}
              max={2000}
              disabled={!!spec.networkVolume}
              value={spec.volumeSize}
              onChange={(e) => set({ volumeSize: Number(e.target.value) })}
            />
          </Field>
        )}
        <Field
          label="Container disk, GB"
          hint="The image and temporary files; emptied when the pod stops."
        >
          <input
            type="number"
            min={20}
            max={500}
            value={spec.disk}
            onChange={(e) => set({ disk: Number(e.target.value) })}
          />
        </Field>
      </div>
      {price !== undefined && (
        <p className="pod-price">
          About <strong>{money(price)} per hour</strong> while it runs, plus
          storage.
          {chosen && chosen.availability === "NONE" && (
            <span className="bad"> None free now: choose another.</span>
          )}
          {chosen &&
            chosen.availability !== "NONE" &&
            spec.dataCenter &&
            !centers.some((d) => d.id === spec.dataCenter) && (
              <span className="bad">
                {" "}
                None free in {centerName(spec.dataCenter)} now.
              </span>
            )}
        </p>
      )}
    </>
  );
}

/** RunPod's GPUs as a table, so memory, price and what is free stand out; a row is chosen by
 * clicking it. Only NVIDIA GPUs can be chosen: the image needs CUDA. */
function GpuTable({
  gpus,
  chosen,
  choose,
}: {
  gpus: RunPodCatalog["gpus"];
  chosen: string;
  choose: (id: string) => void;
}) {
  const cheapest = Math.min(
    ...gpus.filter((g) => usable(g)).map((g) => g.price),
  );
  return (
    <div className="gpu-table">
      <table>
        <thead>
          <tr>
            <th title="Who makes the GPU. The engine image runs on NVIDIA (CUDA) only.">
              Maker
            </th>
            <th title="The GPU model. Click a row to choose it.">GPU</th>
            <th
              className="num"
              title="Video memory (VRAM). The model and its context must fit: about 1 GB per 1B parameters at Q8, half of that at Q4; fine-tuning needs two to three times the model's size."
            >
              Memory
            </th>
            <th
              className="num"
              title="Dollars per hour for one GPU in the chosen cloud, billed by the second while the pod runs; storage is extra. Green: the cheapest GPU free now."
            >
              Price
            </th>
            <th title="How many RunPod has free now, in its own four levels (no exact counts): many, some, few, none. Hover a value for more.">
              Free now
            </th>
          </tr>
        </thead>
        <tbody>
          {gpus.map((g) => {
            const ok = g.manufacturer === "NVIDIA";
            return (
              <tr
                key={g.id}
                className={[
                  g.id === chosen ? "chosen" : "",
                  ok ? "" : "unusable",
                ].join(" ")}
                aria-selected={g.id === chosen}
                title={
                  ok
                    ? undefined
                    : "Cannot be chosen: the engine image runs on NVIDIA (CUDA) only"
                }
                onClick={() => ok && choose(g.id)}
              >
                <td>
                  <span
                    className={`maker maker-${g.manufacturer.toLowerCase()}`}
                  >
                    {maker(g.manufacturer)}
                  </span>
                </td>
                <td
                  className="gpu-name"
                  title={
                    ok ? `RunPod id: ${g.id}. Click to choose.` : undefined
                  }
                >
                  {g.name}
                </td>
                <td className="num">
                  <span
                    className="vram"
                    title={ok ? `${g.memory} GB of video memory` : undefined}
                  >
                    {g.memory} GB
                  </span>
                </td>
                <td className="num">
                  <span
                    className={`price ${usable(g) && g.price === cheapest ? "cheapest" : ""}`}
                    title={
                      ok
                        ? `${money(g.price)} per hour while the pod runs` +
                          (usable(g) && g.price === cheapest
                            ? ": the cheapest GPU free now"
                            : "")
                        : undefined
                    }
                  >
                    {money(g.price)}/h
                  </span>
                </td>
                <td>
                  {ok ? (
                    <span
                      className={`free free-${g.availability.toLowerCase()}`}
                      title={AVAILABILITY_HINT[g.availability]}
                    >
                      {AVAILABILITY[g.availability]}
                    </span>
                  ) : (
                    <span className="muted">needs NVIDIA image</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const usable = (g: RunPodCatalog["gpus"][number]) =>
  g.manufacturer === "NVIDIA" && g.availability !== "NONE";
