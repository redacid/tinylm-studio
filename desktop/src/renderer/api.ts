// Thin client for the engine. Chat goes through the OpenAI-compatible endpoint,
// exactly like an external client would use it.

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type CatalogEntry = {
  id: string;
  repo: string;
  params: string;
  description: string;
  gated: boolean;
  installed: boolean;
  download: { state: "downloading" | "done" | "error"; error: string | null } | null;
};

export type Collection = { name: string; documents: number; chunks: number };

export type DocumentInfo = {
  id: number;
  name: string;
  source: string;
  chars: number;
  chunks: number;
  created_at: string;
};

export type DocumentFilter = {
  query?: string;
  kind?: "all" | "file" | "web";
  sort?: "oldest" | "newest" | "name" | "largest";
  offset?: number;
  limit?: number;
};

export type UploadResult = { name: string; chunks?: number; duplicate?: boolean; error?: string };

export type Source = { index: number; document: string; source: string; text: string; score: number };

export type CrawlSettings = {
  urls: string[];
  /** New pages to download; pages downloaded by earlier crawls are skipped and not counted. */
  max_pages: number;
  max_depth: number;
  per_minute: number;
  use_browser: boolean;
};

/** A saved website: start addresses and crawl settings that can be run again. */
export type Website = CrawlSettings & {
  id: number;
  name: string;
  collection: string;
  last_crawled_at: string | null;
};

export type CrawlJob = {
  id: string;
  collection: string;
  state: "running" | "done" | "cancelled" | "error";
  website_id: number | null;
  fetched: number;
  known: number;
  indexed: number;
  duplicates: number;
  skipped: number;
  failed: number;
  current: string | null;
  log: string[];
  error: string | null;
  options: CrawlSettings;
};

export type SearchKind = "web" | "video";

export type SearchResult = {
  title: string;
  url: string;
  snippet: string;
  kind: SearchKind;
  thumbnail: string | null;
  source: string;
  site: string;
};

/** Per-answer figures reported by the engine (same names as llama.cpp server's `timings`). */
export type Timings = {
  load_ms: number;
  retrieval_ms: number;
  prompt_n: number;
  prompt_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_second: number;
  total_ms: number;
};

export type ChatOptions = {
  signal?: AbortSignal;
  collection?: string;
  /** The engine has loaded the model and searched the data, and starts reading the prompt. */
  onStart?: () => void;
  onSources?: (sources: Source[]) => void;
  onTimings?: (timings: Timings) => void;
};

export class EngineClient {
  constructor(private readonly baseUrl: string) {}

  async installedModels(): Promise<string[]> {
    const body = await this.json<{ data: { id: string }[] }>("/v1/models");
    return body.data.map((m) => m.id);
  }

  /** The engine's recommended model, used when it is installed. */
  async defaultModel(): Promise<string> {
    return (await this.json<{ default: string }>("/api/models")).default;
  }

  async catalog(): Promise<CatalogEntry[]> {
    return (await this.json<{ catalog: CatalogEntry[] }>("/api/models")).catalog;
  }

  async download(modelId: string): Promise<void> {
    await this.json(`/api/models/${encodeURIComponent(modelId)}/download`, { method: "POST" });
  }

  async collections(): Promise<Collection[]> {
    return (await this.json<{ data: Collection[] }>("/api/collections")).data;
  }

  async createCollection(name: string): Promise<void> {
    await this.json("/api/collections", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name }),
    });
  }

  async deleteCollection(name: string): Promise<void> {
    await this.send(`/api/collections/${encodeURIComponent(name)}`, { method: "DELETE" });
  }

  /** One page of a collection's documents, and how many match the filter in total. */
  async documents(collection: string, filter: DocumentFilter = {}): Promise<{ data: DocumentInfo[]; total: number }> {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(filter)) {
      if (value !== undefined && value !== "") params.set(key, String(value));
    }
    return this.json(`/api/collections/${encodeURIComponent(collection)}/documents?${params}`);
  }

  async upload(collection: string, file: File): Promise<UploadResult> {
    const form = new FormData();
    // Folder uploads send the path inside the folder, so documents with the same file name
    // (e.g. index.html of saved web pages) stay distinguishable.
    form.append("files", file, file.webkitRelativePath || file.name);
    const path = `/api/collections/${encodeURIComponent(collection)}/documents`;
    return (await this.json<{ data: UploadResult[] }>(path, { method: "POST", body: form })).data[0];
  }

  async deleteDocument(collection: string, id: number): Promise<void> {
    await this.send(`/api/collections/${encodeURIComponent(collection)}/documents/${id}`, {
      method: "DELETE",
    });
  }

  async startCrawl(collection: string, settings: CrawlSettings): Promise<CrawlJob> {
    return this.json<CrawlJob>(`/api/collections/${encodeURIComponent(collection)}/crawl`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(settings),
    });
  }

  async websites(collection: string): Promise<Website[]> {
    return (await this.json<{ data: Website[] }>(`/api/collections/${encodeURIComponent(collection)}/websites`)).data;
  }

  async addWebsite(collection: string, website: CrawlSettings & { name: string }): Promise<Website> {
    return this.json<Website>(`/api/collections/${encodeURIComponent(collection)}/websites`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(website),
    });
  }

  async updateWebsite(id: number, website: CrawlSettings & { name: string }): Promise<Website> {
    return this.json<Website>(`/api/websites/${id}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(website),
    });
  }

  async deleteWebsite(id: number): Promise<void> {
    await this.send(`/api/websites/${id}`, { method: "DELETE" });
  }

  async crawlWebsite(id: number): Promise<CrawlJob> {
    return this.json<CrawlJob>(`/api/websites/${id}/crawl`, { method: "POST" });
  }

  /** Finds pages or videos on the web with the engine's search provider (SearXNG or Brave). */
  async search(query: string, kind: SearchKind, language = "", limit = 20): Promise<SearchResult[]> {
    const body = await this.json<{ data: SearchResult[] }>("/api/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, kind, language: language || null, limit }),
    });
    return body.data;
  }

  /** Keeps search results (e.g. trailers) as small link documents that answers can cite. */
  async addLinks(collection: string, links: SearchResult[]): Promise<void> {
    await this.json(`/api/collections/${encodeURIComponent(collection)}/links`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ links }),
    });
  }

  async crawls(): Promise<CrawlJob[]> {
    return (await this.json<{ data: CrawlJob[] }>("/api/crawls")).data;
  }

  async cancelCrawl(id: string): Promise<void> {
    await this.send(`/api/crawls/${id}/cancel`, { method: "POST" });
  }

  /**
   * Streams a chat completion, calling onText for every text fragment. With a collection,
   * the engine answers from that uploaded data and reports the sources it used via onSources.
   */
  async chat(
    model: string,
    messages: ChatMessage[],
    onText: (text: string) => void,
    options: ChatOptions = {},
  ) {
    const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, messages, stream: true, max_tokens: 512, collection: options.collection }),
      signal: options.signal,
    });
    if (!response.ok || !response.body) throw new Error(await errorText(response));

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const event of events) {
        const data = event.replace(/^data: /, "");
        if (data === "[DONE]") return;
        const chunk = JSON.parse(data);
        if (chunk.choices?.[0]?.delta?.role) options.onStart?.();
        if (chunk.sources) options.onSources?.(chunk.sources);
        if (chunk.timings) options.onTimings?.(chunk.timings);
        const text = chunk.choices?.[0]?.delta?.content;
        if (text) onText(text);
      }
    }
  }

  private async json<T = unknown>(path: string, init?: RequestInit): Promise<T> {
    return (await this.send(path, init)).json() as Promise<T>;
  }

  private async send(path: string, init?: RequestInit): Promise<Response> {
    const response = await fetch(`${this.baseUrl}${path}`, init);
    if (!response.ok) throw new Error(await errorText(response));
    return response;
  }
}

async function errorText(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = typeof body.detail === "string" ? body.detail : undefined;
    return body.error?.message ?? detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}
