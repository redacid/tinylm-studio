// A small client for RunPod's REST API v2: the one pod that runs the engine image.
// https://docs.runpod.io/api-reference-v2/overview

/** The engine's port inside the pod; RunPod's proxy serves it as https://<pod id>-8765.proxy.runpod.net. */
export const ENGINE_PORT = 8765;

// Overridable for tests, which talk to a fake API and a local engine.
const API_URL = process.env.TINYLM_RUNPOD_API_URL ?? "https://api.runpod.io/v2";
const PROXY_URL = process.env.TINYLM_RUNPOD_PROXY_URL ?? `https://{id}-${ENGINE_PORT}.proxy.runpod.net`;

export type PodStatus = "PROVISIONING" | "STARTING" | "RUNNING" | "EXITED" | "ERROR" | "TERMINATED";

/** What the app shows about the pod. */
export type Pod = {
  id: string;
  name: string;
  status: PodStatus;
  image: string;
  gpu: string | null;
  /** Dollars per hour while it runs. */
  cost: number | null;
  /** How long it has been running, in seconds, while it runs. */
  uptime: number | null;
  /** Where the engine in the pod answers. */
  url: string;
};

/** How a new pod is made. */
export type PodSpec = {
  name: string;
  image: string;
  gpu: string;
  cloud: "SECURE" | "COMMUNITY";
  /** Container disk, in GB: the image and temporary files. */
  disk: number;
  /** A network volume to keep the data on; without one, a pod volume of volumeSize GB. */
  networkVolume: string;
  volumeSize: number;
  env: Record<string, string>;
};

export class RunPodError extends Error {}

export function engineUrl(podId: string): string {
  return PROXY_URL.replace("{id}", podId);
}

export class RunPod {
  constructor(private readonly apiKey: string) {}

  async pod(id: string): Promise<Pod> {
    return this.toPod(await this.call("GET", `/pods/${encodeURIComponent(id)}`));
  }

  async create(spec: PodSpec): Promise<Pod> {
    const mounts = spec.networkVolume
      ? { network: [{ volumeId: spec.networkVolume, path: "/workspace" }] }
      : { persistent: { size: spec.volumeSize, path: "/workspace" } };
    const body = {
      name: spec.name,
      image: spec.image,
      gpu: { id: spec.gpu, count: 1 },
      cloud: spec.cloud,
      disk: spec.disk,
      ports: [`${ENGINE_PORT}/http`],
      mounts,
      env: spec.env,
    };
    return this.toPod(await this.call("POST", "/pods", body));
  }

  async action(id: string, action: "start" | "stop"): Promise<void> {
    await this.call("POST", `/pods/${encodeURIComponent(id)}/action`, { action });
  }

  /** Deletes the pod and its pod volume; a network volume stays. */
  async terminate(id: string): Promise<void> {
    await this.call("DELETE", `/pods/${encodeURIComponent(id)}`);
  }

  private async call(method: string, path: string, body?: unknown): Promise<Record<string, unknown>> {
    let response: Response;
    try {
      response = await fetch(`${API_URL}${path}`, {
        method,
        headers: { authorization: `Bearer ${this.apiKey}`, "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(30_000),
      });
    } catch (error) {
      throw new RunPodError(`RunPod is not reachable: ${(error as Error).message}`);
    }
    const text = await response.text();
    let data: Record<string, unknown> = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      // Not JSON: the text itself is the message.
    }
    if (!response.ok) {
      if (response.status === 401 || response.status === 403)
        throw new RunPodError("RunPod refused the API key: check it in Settings > Engine.");
      // RFC 9457 problem details: title and detail, sometimes a list of field errors.
      const errors = Array.isArray(data.errors) ? data.errors.map((e) => JSON.stringify(e)).join("; ") : "";
      const detail = [data.title, data.detail, errors].filter(Boolean).join(": ") || text || response.statusText;
      throw new RunPodError(`RunPod: ${detail} (HTTP ${response.status})`);
    }
    return data;
  }

  private toPod(data: Record<string, unknown>): Pod {
    const gpu = data.gpu as { id?: string; count?: number } | null | undefined;
    const runtime = data.runtime as { uptime?: number } | null | undefined;
    const id = String(data.id);
    return {
      id,
      name: String(data.name ?? ""),
      status: (data.status as PodStatus) ?? "PROVISIONING",
      image: String(data.image ?? ""),
      gpu: gpu?.id ? (gpu.count && gpu.count > 1 ? `${gpu.count} × ${gpu.id}` : gpu.id) : null,
      cost: typeof data.cost === "number" ? data.cost : null,
      uptime: typeof runtime?.uptime === "number" ? runtime.uptime : null,
      url: engineUrl(id),
    };
  }
}
