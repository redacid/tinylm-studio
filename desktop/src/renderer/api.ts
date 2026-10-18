// Thin client for the engine. Chat goes through the OpenAI-compatible endpoint,
// exactly like an external client would use it.
import type { ModelUpload } from "./types";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type CatalogEntry = {
  id: string;
  repo: string;
  params: string;
  description: string;
  gated: boolean;
  installed: boolean;
  download: Download | null;
};

export type Download = {
  model_id: string;
  repo: string;
  state: "downloading" | "done" | "error";
  error: string | null;
  /** Bytes, when Hugging Face told the sizes. */
  total: number | null;
  downloaded: number | null;
};

/** Per-model settings; a left-out field is the engine's default. */
export type ModelSettings = {
  context_size?: number;
  threads?: number;
  gpu_layers?: number;
  batch_size?: number;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  thinking?: boolean;
  /** Requests that name no collection answer from this one; "" turns off a bundled one. */
  collection?: string;
  rag_top_k?: number;
};

export type InstalledModel = {
  id: string;
  format: "gguf" | "hf" | "adapter";
  /** Bytes on disk. */
  size: number;
  /** The Hugging Face repo (and GGUF file) it was downloaded from; null for a copied folder. */
  source: { repo: string; file?: string | null } | null;
  loaded: boolean;
  settings: ModelSettings;
  /** Fine-tuned models that need this one as their base. */
  used_by: string[];
};

export type ModelInfo = {
  installed: string[];
  models: InstalledModel[];
  downloads: Download[];
  catalog: CatalogEntry[];
  hf_token: boolean;
};

export type HubFormat = "gguf" | "hf";

export type HubModel = {
  repo: string;
  downloads: number;
  likes: number;
  gated: boolean;
  updated: string | null;
  pipeline: string | null;
  params: number | null;
  context_length: number | null;
};

export type HubFiles = {
  repo: string;
  gated: boolean;
  gguf: { file: string; size: number; parts: number; quantization: string | null }[];
  /** The Hugging Face-format weights, or null when the repo has none. */
  hf: { size: number } | null;
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
  /** Follow only links under the start addresses' folders, not the whole site. */
  only_under_start: boolean;
  /** Links never followed: a part of the address, or a pattern with * for the whole address. */
  exclude: string[];
};

/** Options left out come from the crawler settings (Settings > Crawler). */
export type CrawlRequest = Pick<CrawlSettings, "urls"> & Partial<Omit<CrawlSettings, "urls">>;

/** The engine's crawler settings: robots.txt, rate, defaults for new crawls, how pages are fetched. */
export type CrawlerSettings = {
  respect_robots: boolean;
  per_minute: number;
  max_pages: number;
  max_depth: number;
  use_browser: boolean;
  user_agent: string;
  cdp_url: string;
};

/** A saved website: start addresses and crawl settings that can be run again. */
export type Website = CrawlSettings & {
  id: number;
  name: string;
  collection: string;
  last_crawled_at: string | null;
};

/** Adding an exported collection's documents, in the background. */
export type ImportJob = {
  id: string;
  collection: string;
  state: "running" | "done" | "error";
  /** Documents in the file, when its header says. */
  total: number | null;
  added: number;
  duplicates: number;
  error: string | null;
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

export type SearchProviderName = "searxng" | "brave" | "tavily" | "exa" | "perplexity" | "serper";

/** The engine's search settings; API keys are never sent back, only whether each is set. */
export type SearchSettings = {
  provider: SearchProviderName;
  include_domains: string[];
  exclude_domains: string[];
  searxng_url: string;
  brave_api_key_set: boolean;
  tavily_api_key_set: boolean;
  tavily_depth: "basic" | "advanced";
  tavily_topic: "general" | "news";
  exa_api_key_set: boolean;
  exa_type: "auto" | "fast" | "instant";
  perplexity_api_key_set: boolean;
  serper_api_key_set: boolean;
  serper_country: string | null;
};

/** Fields to save; a key is set by a string and removed by "". */
export type SearchSettingsChanges = Partial<
  Omit<SearchSettings, `${string}_api_key_set`> & Record<`${Exclude<SearchProviderName, "searxng">}_api_key`, string>
>;

export type SearchOutcome = {
  provider: SearchProviderName;
  data: SearchResult[];
  /** Results dropped by the domain filter. */
  hidden: number;
  queries: string[];
  include_domains: string[];
  exclude_domains: string[];
};

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

export type TeacherKind = "local" | "openai" | "anthropic" | "cli";

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
  /** Answer settings for every request (chat, generation, judging); null: each request's own. */
  max_tokens: number | null;
  temperature: number | null;
  thinking: ThinkingMode;
  created_at: string | null;
};

/** Thinking of models that have it: the model's default, off or on. */
export type ThinkingMode = "default" | "off" | "on";

export type TeacherInput = {
  /** A saved teacher's id: its key is used when api_key is left empty. */
  id?: number | null;
  name?: string;
  kind: "openai" | "anthropic" | "cli";
  base_url?: string;
  model: string;
  api_key?: string | null;
  per_minute?: number | null;
  max_tokens?: number | null;
  temperature?: number | null;
  thinking?: ThinkingMode;
};

export type TeacherModel = { id: string; name: string; free: boolean };

/** A subscription's command-line tool on the engine (cli teachers) and its login. */
export type CliTool = {
  tool: "claude" | "codex" | "gemini";
  name: string;
  installed: boolean;
  version: string | null;
  logged_in: boolean;
  /** Who is logged in, when the tool says. */
  account: string | null;
  login: CliLogin | null;
};

export type CliLogin = {
  state: "running" | "done" | "failed" | "cancelled";
  /** The tool's output, without terminal codes. */
  output: string;
  /** The link to open in a browser. */
  url: string | null;
  /** Waiting for the code the login page shows (Claude Code, Gemini CLI). */
  wants_code: boolean;
  /** A code was sent and the tool is checking it. */
  checking_code: boolean;
  error: string | null;
};

/** A table SQL queries may read (the SQL tab), with its columns. */
export type DataTable = {
  name: string;
  about: string;
  /** DELETE FROM it is allowed. */
  deletable: boolean;
  columns: { name: string; type: string; not_null: boolean; primary_key: boolean; about: string }[];
};

export type DataSchema = {
  tables: DataTable[];
  examples: { title: string; sql: string }[];
};

export type QueryResult = {
  columns: string[];
  rows: unknown[][];
  /** There were more rows than the limit. */
  truncated: boolean;
  elapsed_ms: number;
  /** A DELETE: the rows it matches, and whether they were deleted (confirm) or only counted. */
  delete?: { table: string; rows: number; done: boolean };
};

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
  /** Put sources found in the collection into each example's system prompt, as the chat does. */
  with_sources: boolean;
};

/** The model and provider that answered a teacher request (OpenRouter may route elsewhere). */
export type Served = { model: string | null; provider: string | null; requests?: number };

/** One request to the teacher during a generation, and what came of it. */
export type TeacherRequest = {
  document: string;
  /** "write": the pairs from a chunk; "check": the pairs against the text. */
  step: "write" | "check";
  /** As the API reported them; the model asked for until an answer says otherwise. */
  model: string | null;
  provider: string | null;
  /** null while the request runs. */
  ok: boolean | null;
  result: string;
  at: string;
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
  /** The teacher requests of the run, newest last (the last 100). */
  requests: TeacherRequest[];
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
  /** Sampling and answer settings; the engine's defaults where left out. */
  params?: ChatParams;
  /** Off: one response with the whole answer instead of a stream. Default on. */
  stream?: boolean;
  /** The engine has loaded the model and searched the data, and starts reading the prompt. */
  onStart?: () => void;
  onSources?: (sources: Source[]) => void;
  onTimings?: (timings: Timings) => void;
  /** Filled in as the exchange goes: the request sent and every chunk received, for debugging. */
  trace?: ChatTrace;
};

export type ChatParams = {
  /** Left out: the model's settings from the Models tab, then the engine's defaults. */
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  repetition_penalty?: number;
  /** How many chunks of the collection to add. */
  rag_top_k?: number;
};

/** One chat request and its streamed response, exactly as sent and received. */
export type ChatTrace = {
  request: Record<string, unknown>;
  status?: number;
  /** The parsed `data:` events of the stream, ending with "[DONE]"; the whole response when
   * not streaming; or the error body. */
  chunks: unknown[];
};

export class EngineClient {
  /** apiKey: the engine's TINYLM_API_KEY, sent with every request when set. */
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey?: string,
  ) {}

  private withKey(init: RequestInit = {}): RequestInit {
    if (!this.apiKey) return init;
    const headers = new Headers(init.headers);
    headers.set("authorization", `Bearer ${this.apiKey}`);
    return { ...init, headers };
  }

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

  /** Installed models, downloads, the catalog and whether downloads send a Hugging Face token. */
  async catalogInfo(): Promise<ModelInfo> {
    return this.json("/api/models");
  }

  async saveModelSettings(modelId: string, settings: ModelSettings): Promise<ModelSettings> {
    return this.json(`/api/models/${encodeURIComponent(modelId)}/settings`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(settings),
    });
  }

  /** The most downloaded models on Hugging Face that match. */
  async searchHub(query: string, format: HubFormat): Promise<HubModel[]> {
    const params = new URLSearchParams({ q: query, format });
    return (await this.json<{ data: HubModel[] }>(`/api/hub/models?${params}`)).data;
  }

  async hubFiles(repo: string): Promise<HubFiles> {
    return this.json(`/api/hub/files?${new URLSearchParams({ repo })}`);
  }

  /** Downloads one GGUF file of a repo or, without a file, its Hugging Face-format weights. */
  async downloadFromHub(repo: string, file?: string): Promise<Download> {
    return this.json("/api/hub/download", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ repo, file: file ?? null }),
    });
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
  /** With a collection, the new model folder takes it along and the model answers from it. */
  async exportModel(modelId: string, quantization: Quantization, name?: string, collection?: string): Promise<ExportJob> {
    return this.json<ExportJob>(`/api/models/${encodeURIComponent(modelId)}/export`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ quantization, name: name || null, collection: collection || null }),
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

  /** The examples as JSONL, for Import file here or on another engine. */
  async exportDataset(dataset: string): Promise<Blob> {
    return (await this.send(`/api/datasets/${encodeURIComponent(dataset)}/export`)).blob();
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
  async testTeacher(teacher: TeacherInput): Promise<{ reply: string; served: Served | null }> {
    return this.json<{ reply: string; served: Served | null }>("/api/teachers/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(teacher),
    });
  }

  async cliTools(): Promise<CliTool[]> {
    return (await this.json<{ data: CliTool[] }>("/api/cli-tools")).data;
  }

  /** Each tool's last login, quick to ask while one goes on. */
  async cliLogins(): Promise<Record<string, CliLogin | null>> {
    return this.json<Record<string, CliLogin | null>>("/api/cli-tools/logins");
  }

  async cliLogin(tool: string): Promise<CliLogin> {
    return this.json<CliLogin>(`/api/cli-tools/${tool}/login`, { method: "POST" });
  }

  async cliLoginCode(tool: string, code: string): Promise<void> {
    await this.send(`/api/cli-tools/${tool}/login/code`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ code }),
    });
  }

  async cliLoginCancel(tool: string): Promise<void> {
    await this.send(`/api/cli-tools/${tool}/login`, { method: "DELETE" });
  }

  async cliLogout(tool: string): Promise<CliTool> {
    return this.json<CliTool>(`/api/cli-tools/${tool}/logout`, { method: "POST" });
  }

  async dataSchema(): Promise<DataSchema> {
    return this.json<DataSchema>("/api/query/schema");
  }

  /** Runs one SELECT over the data, or a DELETE: only counted unless confirm. */
  async query(sql: string, limit: number, confirm = false): Promise<QueryResult> {
    return this.json<QueryResult>("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sql, limit, confirm }),
    });
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

  /** The collection's documents and chunks as a file, for importCollection on another engine. */
  async exportCollection(collection: string): Promise<Blob> {
    return (await this.send(`/api/collections/${encodeURIComponent(collection)}/export`)).blob();
  }

  /** Starts importing an export into the collection it came from (made if missing). */
  async importCollection(file: File): Promise<ImportJob> {
    const form = new FormData();
    form.append("file", file, file.name);
    return this.json("/api/collections/import", { method: "POST", body: form });
  }

  async importJob(id: string): Promise<ImportJob> {
    return this.json(`/api/collections/imports/${encodeURIComponent(id)}`);
  }

  /** A model file the engine is receiving or installing (see window.tinylm.uploadModel). */
  async modelUpload(id: string): Promise<ModelUpload> {
    return this.json(`/api/models/uploads/${encodeURIComponent(id)}`);
  }

  /** The end of the engine's log file, oldest first. */
  async logs(lines = 1000): Promise<{ file: string; lines: string[] }> {
    return this.json(`/api/logs?lines=${lines}`);
  }

  async deleteDocument(collection: string, id: number): Promise<void> {
    await this.send(`/api/collections/${encodeURIComponent(collection)}/documents/${id}`, {
      method: "DELETE",
    });
  }

  async startCrawl(collection: string, settings: CrawlRequest): Promise<CrawlJob> {
    return this.json<CrawlJob>(`/api/collections/${encodeURIComponent(collection)}/crawl`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(settings),
    });
  }

  async websites(collection: string): Promise<Website[]> {
    return (await this.json<{ data: Website[] }>(`/api/collections/${encodeURIComponent(collection)}/websites`)).data;
  }

  async addWebsite(collection: string, website: CrawlRequest & { name: string }): Promise<Website> {
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

  /** Finds pages or videos on the web with the provider and domain filter of the search settings. */
  async search(query: string, kind: SearchKind, language = "", limit = 20): Promise<SearchOutcome> {
    return this.json<SearchOutcome>("/api/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, kind, language: language || null, limit }),
    });
  }

  async crawlerSettings(): Promise<CrawlerSettings> {
    return this.json<CrawlerSettings>("/api/crawl/settings");
  }

  async saveCrawlerSettings(changes: Partial<CrawlerSettings>): Promise<CrawlerSettings> {
    return this.json<CrawlerSettings>("/api/crawl/settings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(changes),
    });
  }

  async searchSettings(): Promise<SearchSettings> {
    return this.json<SearchSettings>("/api/search/settings");
  }

  async saveSearchSettings(changes: SearchSettingsChanges): Promise<SearchSettings> {
    return this.json<SearchSettings>("/api/search/settings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(changes),
    });
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
    const stream = options.stream ?? true;
    const request = {
      model,
      messages,
      stream,
      max_tokens: 512,
      ...options.params,
      collection: options.collection,
      // Asks the engine for the prompt as the model got it, for the trace.
      debug: options.trace ? true : undefined,
    };
    const trace = options.trace ?? { request, chunks: [] };
    trace.request = JSON.parse(JSON.stringify(request)); // without the undefined fields, as sent
    const response = await fetch(
      `${this.baseUrl}/v1/chat/completions`,
      this.withKey({
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
        signal: options.signal,
      }),
    );
    trace.status = response.status;
    if (!response.ok || !response.body) {
      const text = await response.clone().text().catch(() => "");
      trace.chunks.push(parseOrText(text));
      throw new Error(await errorText(response));
    }
    if (!stream) {
      const body = await response.json();
      trace.chunks.push(body);
      options.onStart?.();
      if (body.sources) options.onSources?.(body.sources);
      const text = body.choices?.[0]?.message?.content;
      if (text) onText(text);
      if (body.timings) options.onTimings?.(body.timings);
      return;
    }

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
        if (data === "[DONE]") {
          trace.chunks.push(data);
          return;
        }
        const chunk = JSON.parse(data);
        trace.chunks.push(chunk);
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
    const response = await fetch(`${this.baseUrl}${path}`, this.withKey(init));
    if (!response.ok) throw new Error(await errorText(response));
    return response;
  }
}

function parseOrText(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
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
