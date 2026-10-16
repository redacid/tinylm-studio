// Electron main process: owns the window, the engine connection and the RunPod pod.
import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { apply, Connection, ConnectionChanges, load, newKey, save, view } from "./connection";
import { Engine } from "./engine";
import { log, logFile, tail } from "./log";
import { Progress, saveModel, uploadModel } from "./modelFiles";
import { createRequest, PodGone, RunPod } from "./runpod";

const repoRoot = join(__dirname, "..", "..", "..");
const logger = log("app");
process.on("unhandledRejection", (error) => logger.error(`Unhandled: ${error instanceof Error ? error.stack : error}`));
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
  const before = connection;
  connection = apply(connection, changes);
  if (connection.mode !== before.mode) logger.info(`Engine: ${before.mode} → ${connection.mode}`);
  if (connection.runpod.podId !== before.runpod.podId)
    logger.info(connection.runpod.podId ? `RunPod pod: ${connection.runpod.podId}` : "RunPod pod forgotten");
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
// A pod deleted on runpod.io is forgotten here too, as if terminated from the app.
function forget(error: unknown): null {
  if (!(error instanceof PodGone)) throw error;
  logger.warn(error.message);
  update({ runpod: { podId: "", engineKey: "" } });
  if (connection.mode === "runpod") engine.disconnect(error.message);
  return null;
}
ipcMain.handle("runpod:pod", () =>
  connection.runpod.podId ? runpod().pod(connection.runpod.podId).catch(forget) : null,
);
// The pod's environment: its engine key and, for gated models (Gemma, Llama) to download in the
// pod, the same Hugging Face token as here.
function podEnv(engineKey: string): Record<string, string> {
  const env: Record<string, string> = { TINYLM_API_KEY: engineKey };
  const hfToken = process.env.HF_TOKEN ?? process.env.TINYLM_HF_TOKEN;
  if (hfToken) env.HF_TOKEN = hfToken;
  return env;
}
// The request "Create pod" would send for these settings, with the secrets in env hidden.
ipcMain.handle("runpod:request", (_event, settings: Connection["runpod"], name: string) => {
  const env = Object.fromEntries(
    Object.keys(podEnv("")).map((key) => [
      key,
      key === "TINYLM_API_KEY" ? "<new key, made when the pod is created>" : "<hidden: from this computer>",
    ]),
  );
  return createRequest({ ...settings, name: name || "tinylm-engine", env });
});
ipcMain.handle("runpod:create", async (_event, name: string) => {
  const settings = connection.runpod;
  if (!settings.image) throw new Error("Enter the engine image, e.g. docker.io/you/tinylm-engine:0.1.0-5a0d901-cuda");
  if (settings.kind === "cpu" && !settings.cpu) throw new Error("Choose a CPU type");
  const engineKey = newKey();
  const hardware = settings.kind === "gpu" ? `${settings.gpu}, ${settings.cloud} cloud` : `${settings.cpu}, ${settings.vcpus} vCPU`;
  logger.info(`Creating a RunPod pod: ${settings.image} on ${hardware}${settings.dataCenter ? ` in ${settings.dataCenter}` : ""}`);
  const pod = await runpod().create({ ...settings, name: name || "tinylm-engine", env: podEnv(engineKey) });
  update({ mode: "runpod", runpod: { podId: pod.id, engineKey } });
  engine.connect(connection);
  return pod;
});
function readyEngine() {
  const state = engine.state;
  if (state.phase !== "ready") throw new Error("The engine is not running");
  return state;
}
const sendProgress = (progress: Progress) => window?.webContents.send("models:progress", progress);
// Saves a model of the engine to a file the user picks; null when they cancel.
ipcMain.handle("models:save", async (_event, model: string) => {
  const target = readyEngine();
  const choice = await dialog.showSaveDialog(window!, {
    title: `Save ${model}`,
    defaultPath: join(app.getPath("downloads"), `${model}.tar`),
    filters: [{ name: "Model archive", extensions: ["tar"] }],
  });
  if (choice.canceled || !choice.filePath) return null;
  await saveModel(target, model, choice.filePath, sendProgress);
  return choice.filePath;
});
// Uploads a model file the user picks (a saved model or a .gguf file) to the engine; null when
// they cancel, else the upload to follow until it is installed.
ipcMain.handle("models:upload", async () => {
  const target = readyEngine();
  const choice = await dialog.showOpenDialog(window!, {
    title: "Import a model",
    properties: ["openFile"],
    filters: [{ name: "Model archive or GGUF file", extensions: ["tar", "gguf"] }],
  });
  if (choice.canceled || !choice.filePaths[0]) return null;
  return uploadModel(target, choice.filePaths[0], sendProgress);
});
ipcMain.handle("logs:app", (_event, lines: number) => ({ file: logFile(), lines: tail(lines) }));
ipcMain.handle("logs:open-folder", () => shell.openPath(dirname(logFile())));
ipcMain.handle("runpod:start", async () => {
  logger.info(`Starting the RunPod pod ${connection.runpod.podId}`);
  update({ mode: "runpod" });
  engine.connect(connection); // starts the pod if it is stopped, then waits for the engine
});
ipcMain.handle("runpod:stop", async () => {
  logger.info(`Stopping the RunPod pod ${connection.runpod.podId}`);
  try {
    await runpod().action(connection.runpod.podId, "stop");
  } catch (error) {
    return forget(error);
  }
  if (connection.mode === "runpod") engine.disconnect("The RunPod pod is stopped");
});
ipcMain.handle("runpod:terminate", async () => {
  logger.info(`Terminating the RunPod pod ${connection.runpod.podId}`);
  await runpod()
    .terminate(connection.runpod.podId)
    .catch((error) => error instanceof PodGone || Promise.reject(error));
  update({ runpod: { podId: "", engineKey: "" } });
  if (connection.mode === "runpod") engine.disconnect("The RunPod pod was terminated");
});

app.whenReady().then(() => {
  logger.info(
    `tinylm-studio ${app.getVersion()} started (Electron ${process.versions.electron}, ${process.platform}); settings in ${app.getPath("userData")}`,
  );
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
    if (["PROVISIONING", "STARTING", "RUNNING"].includes(pod.status)) {
      logger.info(`Stopping the RunPod pod ${pod.id} on quit`);
      await runpod.action(pod.id, "stop");
    }
  };
  const timeout = new Promise((resolve) => setTimeout(resolve, 10_000));
  Promise.race([stop(), timeout])
    .catch((error) => logger.error(`Could not stop the RunPod pod: ${error}`))
    .finally(() => app.quit());
});
app.on("window-all-closed", () => app.quit());
