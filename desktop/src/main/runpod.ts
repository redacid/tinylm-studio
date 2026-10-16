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
  /** "NVIDIA GeForce RTX 4090", "2 × NVIDIA A40", or "cpu3c · 8 vCPU". */
  hardware: string | null;
  dataCenter: string | null;
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
  /** A GPU pod (the cuda image), or a CPU-only one (the cpu image). */
  kind: "gpu" | "cpu";
  gpu: string;
  /** CPU flavor and vCPUs, for a CPU pod. */
  cpu: string;
  vcpus: number;
  cloud: "SECURE" | "COMMUNITY";
  /** Data center id, e.g. "EU-RO-1"; "" lets RunPod choose. */
  dataCenter: string;
  /** Container disk, in GB: the image and temporary files. */
  disk: number;
  /** A network volume to keep the data on; without one, a pod volume of volumeSize GB (GPU pods). */
  networkVolume: string;
  volumeSize: number;
  env: Record<string, string>;
};

export type Availability = "NONE" | "LOW" | "MEDIUM" | "HIGH";

/** RunPod's catalog for this account: what a pod can be made of, and where. */
export type Catalog = {
  gpus: {
    id: string;
    name: string;
    /** NVIDIA, AMD or UNKNOWN; the engine image runs on NVIDIA (CUDA) only. */
    manufacturer: string;
    memory: number;
    /** Dollars per hour for one GPU in the chosen cloud. */
    price: number;
    availability: Availability;
    /** Data centers with this GPU free now, and how many. */
    dataCenters: { id: string; availability: Availability }[];
  }[];
  cpus: {
    id: string;
    name: string;
    group: string;
    vcpu: { min: number; max: number };
    ramGbPerVcpu: number;
    /** Dollars per hour per vCPU. */
    pricePerVcpu: number;
    availability: Availability;
    dataCenters: { id: string; availability: Availability }[];
  }[];
  dataCenters: { id: string; name: string; region: string }[];
  volumes: { id: string; name: string; size: number; dataCenter: string }[];
};

/** The engine image is built for CUDA 12.6: hosts with an older driver cannot run it. */
const MIN_CUDA = "12.6";

export class RunPodError extends Error {}

/** The pod is not on RunPod (any more): deleted on runpod.io, for example. */
export class PodGone extends RunPodError {}

export function engineUrl(podId: string): string {
  return PROXY_URL.replace("{id}", podId);
}

/** The body of POST /pods that makes a pod from the spec. */
export function podBody(spec: PodSpec): Record<string, unknown> {
  // CPU pods take no pod volume: without a network volume their data is on the container disk.
  const mounts = spec.networkVolume
    ? { network: [{ volumeId: spec.networkVolume, path: "/workspace" }] }
    : spec.kind === "gpu"
      ? { persistent: { size: spec.volumeSize, path: "/workspace" } }
      : undefined;
  return {
    name: spec.name,
    image: spec.image,
    ...(spec.kind === "gpu"
      ? { gpu: { id: spec.gpu, count: 1, minCudaVersion: MIN_CUDA } }
      : { cpu: { id: spec.cpu, vcpuCount: spec.vcpus } }),
    // CPU pods run in RunPod's own data centers only.
    cloud: spec.kind === "gpu" ? spec.cloud : "SECURE",
    ...(spec.dataCenter ? { dataCenterIds: [spec.dataCenter] } : {}),
    disk: spec.disk,
    ports: [`${ENGINE_PORT}/http`],
    ...(mounts ? { mounts } : {}),
    env: spec.env,
  };
}

/** The request the app sends to RunPod to create a pod. */
export function createRequest(spec: PodSpec): { method: string; url: string; body: Record<string, unknown> } {
  return { method: "POST", url: `${API_URL}/pods`, body: podBody(spec) };
}

export class RunPod {
  constructor(private readonly apiKey: string) {}

  async pod(id: string): Promise<Pod> {
    return this.toPod(await this.call("GET", `/pods/${encodeURIComponent(id)}`));
  }

  async create(spec: PodSpec): Promise<Pod> {
    return this.toPod(await this.call("POST", "/pods", podBody(spec)));
  }

  /** GPUs and CPUs with their prices and where they are free now, data centers and volumes. */
  async catalog(cloud: "SECURE" | "COMMUNITY"): Promise<Catalog> {
    const gpuQuery = new URLSearchParams({ include: "AVAILABILITY", product: "POD", cloud, minCudaVersion: MIN_CUDA });
    const [gpus, cpus, centers, volumes] = await Promise.all([
      this.call("GET", `/catalog/gpus?${gpuQuery}`),
      this.call("GET", "/catalog/cpus?include=AVAILABILITY&product=POD"),
      this.call("GET", "/catalog/datacenters"),
      this.call("GET", "/network-volumes"),
    ]);
    const list = (data: Record<string, unknown>, key: string) =>
      (Array.isArray(data[key]) ? data[key] : []) as Record<string, any>[];
    const places = (item: Record<string, any>) =>
      list(item, "dataCenters").map((d) => ({ id: String(d.id), availability: d.availability as Availability }));
    const secure = cloud === "SECURE";
    return {
      gpus: list(gpus, "gpus")
        .filter((g) => (secure ? g.secure : g.community))
        .map((g) => ({
          id: String(g.id),
          name: String(g.name ?? g.id),
          manufacturer: String(g.manufacturer ?? "UNKNOWN"),
          memory: Number(g.memory ?? 0),
          price: Number((secure ? g.price?.secure : g.price?.community) ?? 0),
          availability: (g.availability ?? "NONE") as Availability,
          dataCenters: places(g),
        })),
      cpus: list(cpus, "cpus").map((c) => ({
        id: String(c.id),
        name: String(c.name ?? c.id),
        group: String(c.group ?? ""),
        vcpu: { min: Number(c.vcpu?.min ?? 2), max: Number(c.vcpu?.max ?? 2) },
        ramGbPerVcpu: Number(c.ramGbPerVcpu ?? 0),
        pricePerVcpu: Number(c.price?.securePerVcpu ?? 0),
        availability: (c.availability ?? "NONE") as Availability,
        dataCenters: places(c),
      })),
      dataCenters: list(centers, "dataCenters").map((d) => ({
        id: String(d.id),
        name: String(d.name ?? d.id),
        region: String(d.region ?? ""),
      })),
      volumes: list(volumes, "networkVolumes").map((v) => ({
        id: String(v.id),
        name: String(v.name ?? v.id),
        size: Number(v.size ?? 0),
        dataCenter: String(v.dataCenter ?? ""),
      })),
    };
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
      if (response.status === 404 && path.startsWith("/pods/"))
        throw new PodGone(`The pod ${decodeURIComponent(path.split("/")[2])} is not on RunPod any more; create a new one`);
      // RFC 9457 problem details: title and detail, sometimes a list of field errors.
      const errors = Array.isArray(data.errors) ? data.errors.map((e) => JSON.stringify(e)).join("; ") : "";
      const detail = [data.title, data.detail, errors].filter(Boolean).join(": ") || text || response.statusText;
      throw new RunPodError(`RunPod: ${detail} (HTTP ${response.status})`);
    }
    return data;
  }

  private toPod(data: Record<string, unknown>): Pod {
    const gpu = data.gpu as { id?: string; count?: number } | null | undefined;
    const cpu = data.cpu as { id?: string; vcpuCount?: number } | null | undefined;
    const runtime = data.runtime as { uptime?: number } | null | undefined;
    const id = String(data.id);
    return {
      id,
      name: String(data.name ?? ""),
      status: (data.status as PodStatus) ?? "PROVISIONING",
      image: String(data.image ?? ""),
      hardware: gpu?.id
        ? gpu.count && gpu.count > 1
          ? `${gpu.count} × ${gpu.id}`
          : gpu.id
        : cpu?.id
          ? `${cpu.id} · ${cpu.vcpuCount ?? "?"} vCPU`
          : null,
      dataCenter: typeof data.dataCenterId === "string" ? data.dataCenterId : null,
      cost: typeof data.cost === "number" ? data.cost : null,
      uptime: typeof runtime?.uptime === "number" ? runtime.uptime : null,
      url: engineUrl(id),
    };
  }
}
