export type EngineState =
  | { phase: "setting-up" }
  | { phase: "starting"; where: string; detail?: string }
  | { phase: "ready"; url: string; apiKey?: string; version: string; commit: string | null; where: string; remote: boolean }
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
    kind: "gpu" | "cpu";
    gpu: string;
    cpu: string;
    vcpus: number;
    cloud: "SECURE" | "COMMUNITY";
    dataCenter: string;
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
  hardware: string | null;
  dataCenter: string | null;
  cost: number | null;
  uptime: number | null;
  url: string;
};

export type Availability = "NONE" | "LOW" | "MEDIUM" | "HIGH";

/** RunPod's catalog: GPUs and CPUs with prices and where they are free now. */
export type RunPodCatalog = {
  gpus: {
    id: string;
    name: string;
    manufacturer: string;
    memory: number;
    price: number;
    availability: Availability;
    dataCenters: { id: string; availability: Availability }[];
  }[];
  cpus: {
    id: string;
    name: string;
    group: string;
    vcpu: { min: number; max: number };
    ramGbPerVcpu: number;
    pricePerVcpu: number;
    availability: Availability;
    dataCenters: { id: string; availability: Availability }[];
  }[];
  dataCenters: { id: string; name: string; region: string }[];
  volumes: { id: string; name: string; size: number; dataCenter: string }[];
};

declare global {
  interface Window {
    tinylm: {
      getEngineState(): Promise<EngineState>;
      onEngineState(callback: (state: EngineState) => void): void;
      getConnection(): Promise<EngineConnection>;
      setConnection(changes: EngineConnectionChanges, reconnect: boolean): Promise<EngineConnection>;
      runpod(action: "catalog", cloud?: "SECURE" | "COMMUNITY"): Promise<RunPodCatalog>;
      runpod(action: "pod"): Promise<Pod | null>;
      runpod(action: "create", name: string): Promise<Pod>;
      runpod(action: "start" | "stop" | "terminate"): Promise<void>;
    };
  }
}
