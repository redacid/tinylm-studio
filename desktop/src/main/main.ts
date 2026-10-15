// Electron main process: owns the window, the engine connection and the RunPod pod.
import { app, BrowserWindow, ipcMain, shell } from "electron";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { apply, Connection, ConnectionChanges, load, newKey, save, view } from "./connection";
import { Engine } from "./engine";
import { RunPod } from "./runpod";

const repoRoot = join(__dirname, "..", "..", "..");
let window: BrowserWindow | undefined;
let connection: Connection;

const engine = new Engine(repoRoot, (state) => window?.webContents.send("engine:state", state));

function createWindow() {
  window = new BrowserWindow({
    width: 1100,
    height: 750,
    title: "tinylm-studio",
    webPreferences: { preload: join(__dirname, "..", "preload", "preload.js") },
  });
  // Links such as a model's Hugging Face page or a chat source open in the system browser; a
  // source file is shown in the file manager rather than opened, so nothing gets run.
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://") || url.startsWith("http://")) {
      shell.openExternal(url);
    } else if (url.startsWith("file://")) {
      const path = fileURLToPath(url);
      if (existsSync(path)) shell.showItemInFolder(path);
    }
    return { action: "deny" };
  });
  window.loadFile(join(__dirname, "..", "renderer", "index.html"));
}

function update(changes: ConnectionChanges): Connection {
  connection = apply(connection, changes);
  save(connection);
  return connection;
}

function runpod(): RunPod {
  if (!connection.runpod.apiKey) throw new Error("Enter your RunPod API key first");
  return new RunPod(connection.runpod.apiKey);
}

ipcMain.handle("engine:get-state", () => engine.state);
ipcMain.handle("engine:get-connection", () => view(connection));
// Saves the changes; reconnect: also switches to the engine they describe.
ipcMain.handle("engine:set-connection", (_event, changes: ConnectionChanges, reconnect: boolean) => {
  update(changes);
  if (reconnect) engine.connect(connection);
  return view(connection);
});
ipcMain.handle("runpod:catalog", (_event, cloud?: "SECURE" | "COMMUNITY") =>
  runpod().catalog(cloud ?? connection.runpod.cloud),
);
ipcMain.handle("runpod:pod", () => (connection.runpod.podId ? runpod().pod(connection.runpod.podId) : null));
ipcMain.handle("runpod:create", async (_event, name: string) => {
  const settings = connection.runpod;
  if (!settings.image) throw new Error("Enter the engine image, e.g. docker.io/you/tinylm-engine:0.1.0-5a0d901-cuda");
  if (settings.kind === "cpu" && !settings.cpu) throw new Error("Choose a CPU type");
  const engineKey = newKey();
  const env: Record<string, string> = { TINYLM_API_KEY: engineKey };
  // Gated models (Gemma, Llama) download in the pod with the same token as here.
  const hfToken = process.env.HF_TOKEN ?? process.env.TINYLM_HF_TOKEN;
  if (hfToken) env.HF_TOKEN = hfToken;
  const pod = await runpod().create({ ...settings, name: name || "tinylm-engine", env });
  update({ mode: "runpod", runpod: { podId: pod.id, engineKey } });
  engine.connect(connection);
  return pod;
});
ipcMain.handle("runpod:start", async () => {
  update({ mode: "runpod" });
  engine.connect(connection); // starts the pod if it is stopped, then waits for the engine
});
ipcMain.handle("runpod:stop", async () => {
  await runpod().action(connection.runpod.podId, "stop");
  if (connection.mode === "runpod") engine.disconnect("The RunPod pod is stopped");
});
ipcMain.handle("runpod:terminate", async () => {
  await runpod().terminate(connection.runpod.podId);
  update({ runpod: { podId: "", engineKey: "" } });
  if (connection.mode === "runpod") engine.disconnect("The RunPod pod was terminated");
});

app.whenReady().then(() => {
  connection = load();
  createWindow();
  engine.connect(connection);
});

// A pod is billed while it runs: stop it on the way out when the settings say so, also when the
// app was switched to another engine since.
let quitting = false;
app.on("before-quit", (event) => {
  engine.stop();
  const settings = connection?.runpod;
  if (quitting || !settings?.stopOnQuit || !settings.podId || !settings.apiKey) return;
  quitting = true;
  event.preventDefault();
  const runpod = new RunPod(settings.apiKey);
  const stop = async () => {
    const pod = await runpod.pod(settings.podId);
    if (["PROVISIONING", "STARTING", "RUNNING"].includes(pod.status)) await runpod.action(pod.id, "stop");
  };
  const timeout = new Promise((resolve) => setTimeout(resolve, 10_000));
  Promise.race([stop(), timeout])
    .catch((error) => console.error("Could not stop the RunPod pod:", error))
    .finally(() => app.quit());
});
app.on("window-all-closed", () => app.quit());
