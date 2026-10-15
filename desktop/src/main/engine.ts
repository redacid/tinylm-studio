// The engine the app talks to: a Python child process on this computer, an engine running
// somewhere else (its address and API key), or one in a RunPod pod that the app starts and stops.
import { ChildProcess, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createServer } from "node:net";
import { homedir } from "node:os";
import { join } from "node:path";
import { Connection } from "./connection";
import { RunPod, RunPodError } from "./runpod";

export type EngineState =
  | { phase: "setting-up" }
  | { phase: "starting"; where: string; detail?: string }
  | {
      phase: "ready";
      url: string;
      apiKey?: string;
      version: string;
      /** The git commit the engine was built from, when it knows. */
      commit: string | null;
      where: string;
      remote: boolean;
    }
  | { phase: "error"; message: string }
  | { phase: "stopped"; message?: string };

const dataDir = join(process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share"), "tinylm-studio");

// How long a pod may take to start and pull the image, and the engine in it to answer.
const POD_TIMEOUT_MS = 20 * 60_000;

// TINYLM_PYTHON selects a user-owned environment; otherwise the managed venv is used.
export function enginePython(): string {
  const venv = process.env.TINYLM_VENV ?? join(dataDir, "venv");
  return process.env.TINYLM_PYTHON ?? join(venv, "bin", "python");
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

function run(command: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: "inherit" });
    child.once("error", reject);
    child.once("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`))));
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

class Cancelled extends Error {}

type Build = { version: string; commit: string | null };

/** The engine's version and commit once /health answers. */
async function waitForHealth(url: string, timeoutMs: number, alive: () => boolean): Promise<Build> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!alive()) throw new Cancelled();
    try {
      const response = await fetch(`${url}/health`, { signal: AbortSignal.timeout(10_000) });
      if (response.ok) {
        const health = (await response.json()) as { version: string; commit?: string | null };
        return { version: health.version, commit: health.commit ?? null };
      }
    } catch {
      // Not listening yet.
    }
    await sleep(url.startsWith("http://127.0.0.1") ? 300 : 3000);
  }
  throw new Error(`The engine at ${url} did not answer in time`);
}

/** Throws when the engine refuses the API key. */
async function checkKey(url: string, apiKey: string): Promise<void> {
  const response = await fetch(`${url}/api/system`, {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {},
    signal: AbortSignal.timeout(15_000),
  });
  if (response.status === 401)
    throw new Error(
      apiKey ? "The engine refused the API key" : "The engine needs an API key (its TINYLM_API_KEY)",
    );
  if (!response.ok) throw new Error(`The engine answered HTTP ${response.status}`);
}

function host(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export class Engine {
  private child?: ChildProcess;
  // Each connect() gets a number; a newer one makes the older give up.
  private attempt = 0;
  state: EngineState = { phase: "stopped" };

  constructor(private readonly repoRoot: string, private readonly onChange: (s: EngineState) => void) {}

  private set(state: EngineState) {
    this.state = state;
    this.onChange(state);
  }

  /** Connects as the settings say, leaving the previous engine (a local one is stopped). */
  async connect(connection: Connection): Promise<void> {
    const attempt = ++this.attempt;
    const alive = () => attempt === this.attempt;
    this.stopLocal();
    try {
      if (connection.mode === "local") await this.startLocal(alive);
      else if (connection.mode === "remote") await this.connectRemote(connection.remote, alive);
      else await this.connectPod(connection.runpod, alive);
    } catch (error) {
      if (error instanceof Cancelled || !alive()) return;
      this.set({ phase: "error", message: (error as Error).message });
    }
  }

  /** Forgets the engine without starting another, e.g. when its pod was stopped. */
  disconnect(message: string): void {
    this.attempt++;
    this.stopLocal();
    this.set({ phase: "stopped", message });
  }

  stop(): void {
    this.attempt++;
    this.state = { phase: "stopped" };
    this.stopLocal();
  }

  private stopLocal() {
    const child = this.child;
    this.child = undefined;
    child?.kill();
  }

  private async startLocal(alive: () => boolean) {
    const python = enginePython();
    if (!existsSync(python)) {
      this.set({ phase: "setting-up" });
      await run("bash", [join(this.repoRoot, "scripts", "setup-engine.sh")]);
    }
    if (!alive()) return;
    const port = await freePort();
    this.set({ phase: "starting", where: `port ${port}` });
    const child = spawn(python, ["-m", "engine", "--port", String(port)], {
      stdio: "inherit",
      // The renderer is loaded from file://; Chromium reports that origin as "null" or "file://".
      // TINYLM_EXIT_WITH_PARENT: the engine stops by itself if this app crashes or is killed.
      env: { ...process.env, TINYLM_CORS_ORIGINS: '["null", "file://"]', TINYLM_EXIT_WITH_PARENT: "1" },
    });
    this.child = child;
    child.once("exit", (code) => {
      if (this.child === child && this.state.phase !== "stopped")
        this.set({ phase: "error", message: `Engine exited with code ${code}` });
    });
    const url = `http://127.0.0.1:${port}`;
    const build = await waitForHealth(url, 30_000, alive);
    if (alive()) this.set({ phase: "ready", url, ...build, where: `port ${port}`, remote: false });
  }

  private async connectRemote(remote: Connection["remote"], alive: () => boolean) {
    const url = remote.url.trim().replace(/\/+$/, "");
    if (!/^https?:\/\//.test(url)) throw new Error("Enter the engine's address in Settings > Engine");
    this.set({ phase: "starting", where: host(url), detail: "Connecting…" });
    const build = await waitForHealth(url, 20_000, alive);
    await checkKey(url, remote.apiKey);
    if (alive()) this.set({ phase: "ready", url, apiKey: remote.apiKey, ...build, where: host(url), remote: true });
  }

  private async connectPod(settings: Connection["runpod"], alive: () => boolean) {
    if (!settings.apiKey) throw new Error("Enter your RunPod API key in Settings > Engine");
    if (!settings.podId) throw new Error("Create a RunPod pod, or enter the id of yours, in Settings > Engine");
    const runpod = new RunPod(settings.apiKey);
    const where = `RunPod ${settings.podId}`;
    const deadline = Date.now() + POD_TIMEOUT_MS;
    let pod = await runpod.pod(settings.podId);
    if (pod.status === "TERMINATED") throw new RunPodError(`The pod ${pod.id} was terminated; create a new one`);
    if (pod.status === "EXITED") {
      this.set({ phase: "starting", where, detail: "Starting the pod…" });
      await runpod.action(pod.id, "start");
      pod = { ...pod, status: "STARTING" };
    }
    while (pod.status !== "RUNNING") {
      if (!alive()) return;
      if (pod.status === "ERROR") throw new RunPodError(`The pod ${pod.id} failed to start; see its logs on runpod.io`);
      if (Date.now() > deadline) throw new RunPodError(`The pod ${pod.id} did not start in time`);
      this.set({ phase: "starting", where, detail: `Pod ${pod.status.toLowerCase()}…` });
      await sleep(5000);
      pod = await runpod.pod(pod.id);
    }
    this.set({ phase: "starting", where, detail: "Waiting for the engine in the pod…" });
    const build = await waitForHealth(pod.url, Math.max(deadline - Date.now(), 60_000), alive);
    await checkKey(pod.url, settings.engineKey);
    if (alive())
      this.set({ phase: "ready", url: pod.url, apiKey: settings.engineKey, ...build, where, remote: true });
  }
}
