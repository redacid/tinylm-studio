// Where the app's engine runs, kept in the app's user data folder. API keys are encrypted with
// the system keychain (Electron's safeStorage) when it is available.
import { app, safeStorage } from "electron";
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

export type Mode = "local" | "remote" | "runpod";

export type Connection = {
  mode: Mode;
  /** An engine somewhere else: its address and the TINYLM_API_KEY it was started with. */
  remote: { url: string; apiKey: string };
  runpod: {
    apiKey: string;
    /** The pod running the engine, once created or entered. */
    podId: string;
    /** The TINYLM_API_KEY given to the pod's engine; generated when the pod is created. */
    engineKey: string;
    // How a new pod is made.
    image: string;
    gpu: string;
    cloud: "SECURE" | "COMMUNITY";
    disk: number;
    networkVolume: string;
    volumeSize: number;
    /** Stop the pod when the app closes, so it is not billed while nobody uses it. */
    stopOnQuit: boolean;
  };
};

/** What the renderer sees: the same, with keys replaced by whether they are set. */
export type ConnectionView = Omit<Connection, "remote" | "runpod"> & {
  remote: { url: string; apiKeySet: boolean };
  runpod: Omit<Connection["runpod"], "apiKey" | "engineKey"> & { apiKeySet: boolean; engineKeySet: boolean };
};

/** Changes from the renderer; a key left out keeps its value, "" clears it. */
export type ConnectionChanges = {
  mode?: Mode;
  remote?: Partial<Connection["remote"]>;
  runpod?: Partial<Connection["runpod"]>;
};

export const DEFAULTS: Connection = {
  mode: "local",
  remote: { url: "", apiKey: "" },
  runpod: {
    apiKey: "",
    podId: "",
    engineKey: "",
    image: "",
    gpu: "NVIDIA GeForce RTX 4090",
    cloud: "SECURE",
    disk: 30,
    networkVolume: "",
    volumeSize: 50,
    stopOnQuit: true,
  },
};

const SECRETS = ["remote.apiKey", "runpod.apiKey", "runpod.engineKey"] as const;

function file(): string {
  return join(app.getPath("userData"), "engine-connection.json");
}

function encrypt(value: string): string {
  if (!value || !safeStorage.isEncryptionAvailable()) return value;
  return `enc:${safeStorage.encryptString(value).toString("base64")}`;
}

function decrypt(value: string): string {
  if (!value.startsWith("enc:")) return value;
  try {
    return safeStorage.decryptString(Buffer.from(value.slice(4), "base64"));
  } catch {
    return ""; // Encrypted on another machine or by another user: has to be entered again.
  }
}

function mapSecrets(connection: Connection, map: (value: string) => string): Connection {
  const copy: Connection = { ...connection, remote: { ...connection.remote }, runpod: { ...connection.runpod } };
  for (const path of SECRETS) {
    const [group, key] = path.split(".") as ["remote" | "runpod", string];
    const target = copy[group] as Record<string, unknown>;
    target[key] = map(String(target[key] ?? ""));
  }
  return copy;
}

export function load(): Connection {
  let saved: Partial<Connection> = {};
  try {
    saved = JSON.parse(readFileSync(file(), "utf8"));
  } catch {
    // Nothing saved yet: the engine runs on this computer.
  }
  const connection: Connection = {
    mode: saved.mode ?? DEFAULTS.mode,
    remote: { ...DEFAULTS.remote, ...saved.remote },
    runpod: { ...DEFAULTS.runpod, ...saved.runpod },
  };
  return mapSecrets(connection, decrypt);
}

export function save(connection: Connection): void {
  const path = file();
  if (!existsSync(dirname(path))) mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(mapSecrets(connection, encrypt), null, 2), { mode: 0o600 });
}

export function apply(connection: Connection, changes: ConnectionChanges): Connection {
  return {
    mode: changes.mode ?? connection.mode,
    remote: { ...connection.remote, ...changes.remote },
    runpod: { ...connection.runpod, ...changes.runpod },
  };
}

export function view(connection: Connection): ConnectionView {
  const { apiKey, engineKey, ...runpod } = connection.runpod;
  return {
    mode: connection.mode,
    remote: { url: connection.remote.url, apiKeySet: !!connection.remote.apiKey },
    runpod: { ...runpod, apiKeySet: !!apiKey, engineKeySet: !!engineKey },
  };
}

export function newKey(): string {
  return randomBytes(24).toString("base64url");
}
