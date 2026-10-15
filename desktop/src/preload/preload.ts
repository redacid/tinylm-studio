// Exposes a minimal, typed bridge from the renderer to the main process.
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("tinylm", {
  getEngineState: () => ipcRenderer.invoke("engine:get-state"),
  onEngineState: (callback: (state: unknown) => void) =>
    ipcRenderer.on("engine:state", (_event, state) => callback(state)),
  getConnection: () => ipcRenderer.invoke("engine:get-connection"),
  setConnection: (changes: unknown, reconnect: boolean) => ipcRenderer.invoke("engine:set-connection", changes, reconnect),
  runpod: (action: "pod" | "create" | "start" | "stop" | "terminate", name?: string) =>
    ipcRenderer.invoke(`runpod:${action}`, name),
});
