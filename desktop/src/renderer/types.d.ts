export type EngineState =
  | { phase: "setting-up" }
  | { phase: "starting"; where: string; detail?: string }
  | { phase: "ready"; url: string; apiKey?: string; version: string; where: string; remote: boolean }
  | { phase: "error"; message: string }
  | { phase: "stopped"; message?: string };

export type EngineMode = "local" | "remote" | "runpod";

/** Where the engine runs, as the app keeps it; API keys only say whether they are set. */
export type EngineConnection = {
  mode: EngineMode;
  remote: { url: string; apiKeySet: boolean };
  runpod: {
    apiKeySet: boolean;
    podId: string;
    engineKeySet: boolean;
    image: string;
    gpu: string;
    cloud: "SECURE" | "COMMUNITY";
    disk: number;
    networkVolume: string;
    volumeSize: number;
    stopOnQuit: boolean;
  };
};

/** Changes to save; a key left out keeps its value. */
export type EngineConnectionChanges = {
  mode?: EngineMode;
  remote?: { url?: string; apiKey?: string };
  runpod?: Partial<Omit<EngineConnection["runpod"], "apiKeySet" | "engineKeySet">> & {
    apiKey?: string;
    engineKey?: string;
  };
};

export type PodStatus = "PROVISIONING" | "STARTING" | "RUNNING" | "EXITED" | "ERROR" | "TERMINATED";

export type Pod = {
  id: string;
  name: string;
  status: PodStatus;
  image: string;
  gpu: string | null;
  cost: number | null;
  uptime: number | null;
  url: string;
};

declare global {
  interface Window {
    tinylm: {
      getEngineState(): Promise<EngineState>;
      onEngineState(callback: (state: EngineState) => void): void;
      getConnection(): Promise<EngineConnection>;
      setConnection(changes: EngineConnectionChanges, reconnect: boolean): Promise<EngineConnection>;
      runpod(action: "pod"): Promise<Pod | null>;
      runpod(action: "create", name: string): Promise<Pod>;
      runpod(action: "start" | "stop" | "terminate"): Promise<void>;
    };
  }
}
