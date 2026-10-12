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
  /** What happened to each start address (without #fragment); missing ones were not reached. */
  outcomes: Record<string, "added" | "duplicate" | "known" | "skipped" | "failed">;
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

export type Dataset = {
  name: string;
  examples: number;
  train: number;
  eval: number;
  /** Share of examples held out to measure the model on data it was not trained on. */
  eval_percent: number;
  created_at: string;
};

export type Example = {
  id: number;
  messages: ChatMessage[];
  split: "train" | "eval";
  origin: ExampleOrigin;
  /** For generated examples: the document they were written from. */
  source: string | null;
  created_at: string;
};

export type ExampleOrigin = "manual" | "imported" | "generated";

export type ExampleFilter = {
  query?: string;
  split?: "all" | "train" | "eval";
  origin?: ExampleOrigin;
  offset?: number;
  limit?: number;
};

export type TeacherKind = "local" | "openai" | "anthropic";

/** A model that writes training examples: a saved API provider or an installed model. */
export type Teacher = {
  id: number | null;
  /** What generation takes: a saved teacher's id or "local:<model id>". */
  ref: string;
  name: string;
  kind: TeacherKind;
  base_url: string;
  model: string;
  has_key: boolean;
  per_minute: number | null;
  created_at: string | null;
};

export type TeacherInput = {
  /** A saved teacher's id: its key is used when api_key is left empty. */
  id?: number | null;
  name?: string;
  kind: "openai" | "anthropic";
  base_url?: string;
  model: string;
  api_key?: string | null;
  per_minute?: number | null;
};

export type TeacherModel = { id: string; name: string; free: boolean };

export type GenerationOptions = {
  collection: string;
  teacher: string;
  chunks: number;
  per_chunk: number;
  languages: string[];
  instructions: string;
  system_prompt: string;
  unanswerable: boolean;
  verify: boolean;
};

export type GenerationJob = {
  id: string;
  dataset: string;
  options: GenerationOptions;
  state: "running" | "done" | "cancelled" | "error";
  chunks_total: number;
  chunks_done: number;
  generated: number;
  added: number;
  duplicates: number;
  /** Pairs dropped, by reason. */
  rejected: Record<string, number>;
  failed_chunks: number;
  current: string | null;
  /** Seconds of a rate-limit pause in progress. */
  waiting: number | null;
  log: string[];
  error: string | null;
  created_at: string;
};

export type ImportResult = {
  name: string;
  added?: number;
  duplicates?: number;
  invalid?: number;
  errors?: { line: number; error: string }[];
  error?: string;
};

export type EvaluationOptions = {
  dataset: string;
  models: string[];
  /** A teacher's ref; null scores by similarity only. */
  judge: string | null;
  limit: number;
  max_tokens: number;
};

export type EvaluationAnswer = {
  text: string;
  seconds: number;
  /** chrF against the reference, 0-100. */
  similarity: number;
  /** The judge's score, 1-5. */
  score: number | null;
  reason: string | null;
};

export type EvaluationSummary = {
  answers: number;
  similarity: number | null;
  judged: number;
  score: number | null;
  seconds: number | null;
  chars: number | null;
};

export type Evaluation = {
  id: string;
  options: EvaluationOptions;
  state: "running" | "done" | "cancelled" | "error" | "interrupted";
  created_at: string;
  finished_at: string | null;
  error: string | null;
  phase: "answering" | "judging" | "";
  current: string | null;
  done: number;
  total: number;
  summary: Record<string, EvaluationSummary>;
  /** Only in a single report. */
  examples?: {
    messages: ChatMessage[];
    reference: string;
    answers: Record<string, EvaluationAnswer>;
    judge_error?: string;
  }[];
};

export type TrainingSettings = {
  /** The new model's name. */
  name: string;
  base: string;
  dataset: string;
  epochs: number;
  learning_rate: number;
  lora_rank: number;
  max_length: number;
  batch_size: number;
  gradient_accumulation: number;
  device: "auto" | "cpu" | "cuda" | "mps";
};

export type LossPoint = { step: number; loss?: number; eval_loss?: number };

export type TrainingJob = {
  id: string;
  settings: TrainingSettings;
  state: "running" | "stopping" | "stopped" | "done" | "error" | "interrupted";
  created_at: string;
  finished_at: string | null;
  error: string | null;
  step: number;
  total_steps: number | null;
  epoch: number;
  loss: number | null;
  eval_loss: number | null;
  device: string | null;
  train_examples: number | null;
  eval_examples: number | null;
  /** Examples left out because their answer starts beyond max_length tokens. */
  dropped: number;
  checkpoint_step: number | null;
  resumable: boolean;
  eta_seconds: number | null;
  history?: LossPoint[];
  log?: string;
};

/** One event of a training run's progress stream. */
export type TrainingEvent =
  | { type: "start"; step: number; total_steps: number; device: string; train_examples: number; eval_examples: number }
  | { type: "step"; step: number; epoch: number; loss: number; learning_rate: number }
  | { type: "eval"; step: number; eval_loss: number }
  | { type: "checkpoint" | "stopped" | "done"; step: number }
  | { type: "error"; message: string }
  | { type: "state"; state: TrainingJob["state"]; error: string | null };

export type Quantization = "Q8_0" | "Q4_K_M" | "F16";

export type ExportJob = {
  id: string;
  model: string;
  name: string;
  quantization: Quantization;
  state: "running" | "done" | "error" | "interrupted";
  step: string | null;
  steps: string[];
  notes: string[];
  error: string | null;
  size: number | null;
  created_at: string;
  finished_at: string | null;
};

export type TrainedModel = {
  id: string;
  /** "adapter": LoRA weights that need the base model; "gguf": a standalone export for llama.cpp. */
  format: "adapter" | "gguf";
  quantization?: string;
  exported_from?: string;
  base: string;
  base_installed: boolean;
  dataset: string;
  created_at: string;
  loss: number | null;
  eval_loss: number | null;
  settings: Record<string, number>;
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
    return (await this.catalogInfo()).catalog;
  }

  /** The catalog and whether downloads send a Hugging Face token. */
  async catalogInfo(): Promise<{ catalog: CatalogEntry[]; hf_token: boolean }> {
    return this.json("/api/models");
  }

  async download(modelId: string): Promise<void> {
    await this.json(`/api/models/${encodeURIComponent(modelId)}/download`, { method: "POST" });
  }

  /** Installed models that can be fine-tuned (PyTorch weights). */
  async trainableModels(): Promise<string[]> {
    return (await this.json<{ trainable: string[] }>("/api/models")).trainable;
  }

  async trainedModels(): Promise<TrainedModel[]> {
    return (await this.json<{ data: TrainedModel[] }>("/api/models/trained")).data;
  }

  /** Merges a fine-tuned model into its base model and writes one GGUF file for llama.cpp. */
  async exportModel(modelId: string, quantization: Quantization, name?: string): Promise<ExportJob> {
    return this.json<ExportJob>(`/api/models/${encodeURIComponent(modelId)}/export`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ quantization, name: name || null }),
    });
  }

  async exports(): Promise<ExportJob[]> {
    return (await this.json<{ data: ExportJob[] }>("/api/exports")).data;
  }

  async deleteModel(modelId: string): Promise<void> {
    await this.send(`/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
  }

  async datasets(): Promise<Dataset[]> {
    return (await this.json<{ data: Dataset[] }>("/api/datasets")).data;
  }

  async createDataset(name: string, evalPercent: number): Promise<Dataset> {
    return this.json<Dataset>("/api/datasets", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, eval_percent: evalPercent }),
    });
  }

  async updateDataset(name: string, evalPercent: number): Promise<Dataset> {
    return this.json<Dataset>(`/api/datasets/${encodeURIComponent(name)}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ eval_percent: evalPercent }),
    });
  }

  async deleteDataset(name: string): Promise<void> {
    await this.send(`/api/datasets/${encodeURIComponent(name)}`, { method: "DELETE" });
  }

  /** Adds the examples of a JSONL, JSON or CSV file; bad records come back with line numbers. */
  async importExamples(dataset: string, file: File): Promise<ImportResult> {
    const form = new FormData();
    form.append("files", file, file.name);
    const path = `/api/datasets/${encodeURIComponent(dataset)}/import`;
    return (await this.json<{ data: ImportResult[] }>(path, { method: "POST", body: form })).data[0];
  }

  async examples(dataset: string, filter: ExampleFilter = {}): Promise<{ data: Example[]; total: number }> {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(filter)) {
      if (value !== undefined && value !== "") params.set(key, String(value));
    }
    return this.json(`/api/datasets/${encodeURIComponent(dataset)}/examples?${params}`);
  }

  async addExample(dataset: string, messages: ChatMessage[]): Promise<{ id: number | null; duplicate: boolean }> {
    return this.json(`/api/datasets/${encodeURIComponent(dataset)}/examples`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages }),
    });
  }

  async editExample(dataset: string, id: number, messages: ChatMessage[]): Promise<void> {
    await this.send(`/api/datasets/${encodeURIComponent(dataset)}/examples/${id}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages }),
    });
  }

  async deleteExample(dataset: string, id: number): Promise<void> {
    await this.send(`/api/datasets/${encodeURIComponent(dataset)}/examples/${id}`, { method: "DELETE" });
  }

  /** Starts writing examples from a collection's documents with a teacher model. */
  async generate(dataset: string, options: GenerationOptions): Promise<GenerationJob> {
    return this.json<GenerationJob>(`/api/datasets/${encodeURIComponent(dataset)}/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(options),
    });
  }

  /** How many chunks of a collection the dataset has not been generated from yet. */
  async unusedChunks(dataset: string, collection: string): Promise<number> {
    const path = `/api/datasets/${encodeURIComponent(dataset)}/generate?collection=${encodeURIComponent(collection)}`;
    return (await this.json<{ unused_chunks: number }>(path)).unused_chunks;
  }

  async generations(): Promise<GenerationJob[]> {
    return (await this.json<{ data: GenerationJob[] }>("/api/generations")).data;
  }

  async generation(id: string): Promise<GenerationJob> {
    return this.json<GenerationJob>(`/api/generations/${id}`);
  }

  async cancelGeneration(id: string): Promise<void> {
    await this.send(`/api/generations/${id}/cancel`, { method: "POST" });
  }

  async teachers(): Promise<Teacher[]> {
    return (await this.json<{ data: Teacher[] }>("/api/teachers")).data;
  }

  async saveTeacher(teacher: TeacherInput): Promise<Teacher> {
    const id = teacher.id;
    return this.json<Teacher>(id == null ? "/api/teachers" : `/api/teachers/${id}`, {
      method: id == null ? "POST" : "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(teacher),
    });
  }

  async deleteTeacher(id: number): Promise<void> {
    await this.send(`/api/teachers/${id}`, { method: "DELETE" });
  }

  /** The provider's models, free ones first. */
  async teacherModels(teacher: TeacherInput): Promise<TeacherModel[]> {
    return (
      await this.json<{ data: TeacherModel[] }>("/api/teachers/models", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(teacher),
      })
    ).data;
  }

  /** Sends a short request; returns the model's reply or throws the provider's error. */
  async testTeacher(teacher: TeacherInput): Promise<string> {
    return (
      await this.json<{ reply: string }>("/api/teachers/test", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(teacher),
      })
    ).reply;
  }

  async evaluations(): Promise<Evaluation[]> {
    return (await this.json<{ data: Evaluation[] }>("/api/evaluations")).data;
  }

  async evaluation(id: string): Promise<Evaluation> {
    return this.json<Evaluation>(`/api/evaluations/${id}`);
  }

  async startEvaluation(options: EvaluationOptions): Promise<Evaluation> {
    return this.json<Evaluation>("/api/evaluations", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(options),
    });
  }

  async stopEvaluation(id: string): Promise<void> {
    await this.send(`/api/evaluations/${id}/stop`, { method: "POST" });
  }

  async deleteEvaluation(id: string): Promise<void> {
    await this.send(`/api/evaluations/${id}`, { method: "DELETE" });
  }

  async trainingJobs(): Promise<TrainingJob[]> {
    return (await this.json<{ data: TrainingJob[] }>("/api/training/jobs")).data;
  }

  /** A run with its full loss history (and the worker's log when it failed). */
  async trainingJob(id: string): Promise<TrainingJob> {
    return this.json<TrainingJob>(`/api/training/jobs/${id}`);
  }

  async startTraining(settings: TrainingSettings): Promise<TrainingJob> {
    return this.json<TrainingJob>("/api/training/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(settings),
    });
  }

  async stopTraining(id: string): Promise<void> {
    await this.send(`/api/training/jobs/${id}/stop`, { method: "POST" });
  }

  async resumeTraining(id: string): Promise<void> {
    await this.send(`/api/training/jobs/${id}/resume`, { method: "POST" });
  }

  async deleteTrainingJob(id: string): Promise<void> {
    await this.send(`/api/training/jobs/${id}`, { method: "DELETE" });
  }

  /** Follows a training run's progress (server-sent events) until it ends or signal aborts. */
  async trainingEvents(id: string, onEvent: (event: TrainingEvent) => void, signal?: AbortSignal) {
    const response = await this.send(`/api/training/jobs/${id}/events`, { signal });
    const reader = response.body!.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += value;
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const event of events) onEvent(JSON.parse(event.replace(/^data: /, "")));
    }
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
  async addLinks(collection: string, links: SearchResult[]): Promise<UploadResult[]> {
    const path = `/api/collections/${encodeURIComponent(collection)}/links`;
    const body = await this.json<{ data: UploadResult[] }>(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ links }),
    });
    return body.data;
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
