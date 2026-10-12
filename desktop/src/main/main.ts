// Electron main process: owns the window and the engine lifecycle.
import { app, BrowserWindow, ipcMain, shell } from "electron";
import { join } from "node:path";
import { Engine } from "./engine";

const repoRoot = join(__dirname, "..", "..", "..");
let window: BrowserWindow | undefined;

const engine = new Engine(repoRoot, (state) => window?.webContents.send("engine:state", state));

function createWindow() {
  window = new BrowserWindow({
    width: 1100,
    height: 750,
    title: "tinylm-studio",
    webPreferences: { preload: join(__dirname, "..", "preload", "preload.js") },
  });
  // Links such as a model's Hugging Face page open in the system browser.
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) shell.openExternal(url);
    return { action: "deny" };
  });
  window.loadFile(join(__dirname, "..", "renderer", "index.html"));
}

ipcMain.handle("engine:get-state", () => engine.state);

app.whenReady().then(() => {
  createWindow();
  engine.start();
});

app.on("before-quit", () => engine.stop());
app.on("window-all-closed", () => app.quit());
