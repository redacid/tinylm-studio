// Exposes a minimal, typed bridge from the renderer to the main process.
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("tinylm", {
  getEngineState: () => ipcRenderer.invoke("engine:get-state"),
  onEngineState: (callback: (state: unknown) => void) =>
    ipcRenderer.on("engine:state", (_event, state) => callback(state)),
  getConnection: () => ipcRenderer.invoke("engine:get-connection"),
  setConnection: (changes: unknown, reconnect: boolean) => ipcRenderer.invoke("engine:set-connection", changes, reconnect),
  // create: the new pod's name; catalog: the cloud; request: the settings and the name.
  runpod: (action: "catalog" | "pod" | "create" | "request" | "start" | "stop" | "terminate", ...args: unknown[]) =>
    ipcRenderer.invoke(`runpod:${action}`, ...args),
});
