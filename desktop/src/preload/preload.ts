// Exposes a minimal, typed bridge from the renderer to the main process.
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("tinylm", {
  getEngineState: () => ipcRenderer.invoke("engine:get-state"),
  onEngineState: (callback: (state: unknown) => void) =>
    ipcRenderer.on("engine:state", (_event, state) => callback(state)),
});
