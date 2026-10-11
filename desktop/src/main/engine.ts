// Starts the Python engine as a child process and tracks its state.
import { ChildProcess, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createServer } from "node:net";
import { homedir } from "node:os";
import { join } from "node:path";

export type EngineState =
  | { phase: "setting-up" }
  | { phase: "starting"; port: number }
  | { phase: "ready"; port: number; version: string }
  | { phase: "error"; message: string }
  | { phase: "stopped" };

const dataDir = join(process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share"), "tinylm-studio");

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

async function waitForHealth(port: number, timeoutMs = 30_000): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      if (response.ok) return ((await response.json()) as { version: string }).version;
    } catch {
      // Engine is not listening yet.
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error("Engine did not become healthy in time");
}

export class Engine {
  private child?: ChildProcess;
  state: EngineState = { phase: "stopped" };

  constructor(private readonly repoRoot: string, private readonly onChange: (s: EngineState) => void) {}

  private set(state: EngineState) {
    this.state = state;
    this.onChange(state);
  }

  async start(): Promise<void> {
    try {
      const python = enginePython();
      if (!existsSync(python)) {
        this.set({ phase: "setting-up" });
        await run("bash", [join(this.repoRoot, "scripts", "setup-engine.sh")]);
      }
      const port = await freePort();
      this.set({ phase: "starting", port });
      this.child = spawn(python, ["-m", "engine", "--port", String(port)], {
        stdio: "inherit",
        // The renderer is loaded from file://; Chromium reports that origin as "null" or "file://".
        env: { ...process.env, TINYLM_CORS_ORIGINS: '["null", "file://"]' },
      });
      this.child.once("exit", (code) => {
        if (this.state.phase !== "stopped") this.set({ phase: "error", message: `Engine exited with code ${code}` });
      });
      const version = await waitForHealth(port);
      this.set({ phase: "ready", port, version });
    } catch (error) {
      this.set({ phase: "error", message: (error as Error).message });
    }
  }

  stop(): void {
    this.state = { phase: "stopped" };
    this.child?.kill();
  }
}
