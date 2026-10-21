"""Response bodies of the API, for the OpenAPI schema at /openapi.json (and the desktop app's
TypeScript types generated from it).

Handlers return plain dicts; FastAPI checks them against these models and sends exactly the
fields a handler returned (see Router). Tests set STRICT, which makes a field the engine sends
but a model leaves out an error, so the schema cannot fall behind the code unnoticed.
"""

from dataclasses import is_dataclass
from functools import cache
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, create_model, model_validator

from engine.datasets.generate import GenerationOptions
from engine.datasets.store import Origin
from engine.evaluation.evaluate import EvaluationOptions
from engine.models.settings import ModelSettings
from engine.training.jobs import TrainingSettings
from engine.web.crawler import CrawlOptions, Outcome
from engine.web.search import SearchKind
from engine.web.search_settings import Provider

# Set by the tests: unknown fields in a response are an error instead of being dropped.
STRICT = False


class Schema(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def known_fields(cls, data: Any) -> Any:
        if STRICT and isinstance(data, dict):
            unknown = set(data) - set(cls.model_fields)
            if unknown:
                raise ValueError(f"{cls.__name__} has no field {', '.join(sorted(unknown))}")
        return data


class Router(APIRouter):
    """An APIRouter whose responses carry exactly the fields a handler returned: a field a
    model has but the handler left out stays out instead of becoming null."""

    def add_api_route(self, path: str, endpoint, **kwargs) -> None:
        kwargs["response_model_exclude_unset"] = True
        super().add_api_route(path, endpoint, **kwargs)


def sent_whole(cls: type) -> type:
    """Marks a dataclass the API sends with all its fields (asdict), so the schema lists
    them all as present, also those with a default."""
    assert is_dataclass(cls)
    cls.__pydantic_config__ = ConfigDict(json_schema_serialization_defaults_required=True)
    return cls


for options in (CrawlOptions, EvaluationOptions, GenerationOptions, TrainingSettings):
    sent_whole(options)


@cache
def items(cls: type) -> type[Schema]:
    """{"data": [...]}, named <cls>List in the schema."""
    return create_model(f"{cls.__name__}List", __base__=Schema, data=list[cls])


@cache
def page(cls: type) -> type[Schema]:
    """One page of a list: {"data": [...], "total": the number of items that match}."""
    return create_model(f"{cls.__name__}Page", __base__=Schema, data=list[cls], total=int)


# Errors


class Error(BaseModel):
    """An error of the management API (/api/...)."""

    detail: str


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class OpenAIError(BaseModel):
    """An error of /v1/chat/completions and /v1/models, and a refused API key, as OpenAI's API
    reports them (Anthropic's form for /v1/messages and requests with anthropic-version)."""

    error: OpenAIErrorDetail


class AnthropicErrorDetail(BaseModel):
    type: str
    message: str


class AnthropicError(BaseModel):
    """An error of /v1/messages, as Anthropic's API reports it."""

    type: Literal["error"]
    error: AnthropicErrorDetail


ERROR_DESCRIPTIONS = {
    400: "Bad request",
    401: "Missing or wrong API key (TINYLM_API_KEY)",
    404: "Not found",
    409: "Conflict with the current state, such as a job that is already running",
    422: "Invalid request",
    502: "The upstream service (Hugging Face, a teacher's provider) failed",
    503: "The search provider is not available",
}


def errors(*codes: int) -> dict[int | str, dict]:
    """The `responses` of an endpoint that can answer with these error statuses. 422 keeps
    FastAPI's own schema: the detail is a string for the engine's checks, a list for invalid
    fields."""
    return {
        code: {"description": ERROR_DESCRIPTIONS[code]}
        if code == 422
        else {"model": Error, "description": ERROR_DESCRIPTIONS[code]}
        for code in codes
    }


UNAUTHORIZED: dict[int | str, dict] = {
    401: {"model": OpenAIError, "description": ERROR_DESCRIPTIONS[401]}
}


def file_response(media_type: str, description: str) -> dict[int | str, dict]:
    return {
        200: {
            "description": description,
            "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
        }
    }


def event_stream(description: str) -> dict[int | str, dict]:
    return {
        200: {
            "description": description,
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    }


# System


class Health(Schema):
    status: Literal["ok"]
    version: str
    commit: str | None


class EngineLog(Schema):
    file: str
    lines: list[str]


class Hardware(Schema):
    cpu: str
    cpu_cores: int
    memory_gb: float
    accelerators: list[str]


class SystemInfo(Schema):
    version: str
    commit: str | None
    python: str
    hardware: Hardware


# Models


class Download(Schema):
    model_id: str
    repo: str
    state: Literal["downloading", "done", "error"]
    error: str | None
    # Bytes, when Hugging Face told the sizes.
    total: int | None
    downloaded: int | None


class CatalogEntry(Schema):
    id: str
    repo: str
    params: str
    description: str
    format: Literal["hf", "gguf"]
    file_pattern: str | None
    gated: bool
    installed: bool
    download: Download | None


class ModelSource(Schema):
    """The Hugging Face repo (and GGUF file) a model was downloaded from."""

    repo: str
    file: str | None = None


class InstalledModel(Schema):
    id: str
    format: Literal["gguf", "hf", "adapter"]
    # Bytes on disk.
    size: int
    # None for a folder copied in by hand.
    source: ModelSource | None
    loaded: bool
    settings: ModelSettings
    # Fine-tuned models that need this one as their base.
    used_by: list[str]


class ModelInfo(Schema):
    # Models that can answer.
    installed: list[str]
    models: list[InstalledModel]
    downloads: list[Download]
    # Installed models that can be fine-tuned (PyTorch weights).
    trainable: list[str]
    catalog: list[CatalogEntry]
    # The recommended model, used when it is installed.
    default: str
    # Whether downloads send a Hugging Face token.
    hf_token: bool


class TrainedModel(Schema):
    id: str
    # "adapter": LoRA weights that need the base model; "gguf": a standalone export.
    format: Literal["adapter", "gguf"]
    base: str
    base_installed: bool
    # From the model's tinylm.json, which a model folder copied in by hand may lack.
    dataset: str | None = None
    job: str | None = None
    created_at: str | None = None
    loss: float | None = None
    eval_loss: float | None = None
    settings: dict[str, int | float] | None = None
    # Exports only:
    quantization: str | None = None
    exported_from: str | None = None
    exported_at: str | None = None
    # The collection the export takes along.
    collection: str | None = None


class ModelUpload(Schema):
    """A model file being received and installed."""

    id: str
    name: str
    size: int
    state: Literal["receiving", "installing", "done", "error"]
    received: int
    model: str | None
    # For a fine-tuned model: its base model, when this engine does not have it.
    missing_base: str | None
    error: str | None
    started_at: str


class HubModel(Schema):
    repo: str
    downloads: int
    likes: int
    gated: bool
    updated: str | None
    pipeline: str | None
    params: int | None
    context_length: int | None


class HubFile(Schema):
    """A GGUF file of a repo; a split file is one entry, named by its first part."""

    file: str
    size: int
    parts: int
    quantization: str | None


class HubWeights(Schema):
    size: int


class HubFiles(Schema):
    repo: str
    gated: bool
    gguf: list[HubFile]
    # The Hugging Face-format weights, or None when the repo has none.
    hf: HubWeights | None


# Data


class Collection(Schema):
    name: str
    created_at: str
    documents: int
    chunks: int


class DocumentInfo(Schema):
    id: int
    name: str
    source: str
    chars: int
    chunks: int
    created_at: str


class UploadResult(Schema):
    """A file or link added to a collection: its document, or why it was not added."""

    name: str
    id: int | None = None
    chunks: int | None = None
    # The collection already has this text.
    duplicate: bool | None = None
    error: str | None = None


class ImportJob(Schema):
    """Adding an exported collection's documents, in the background."""

    id: str
    collection: str
    state: Literal["running", "done", "error"]
    # Documents in the file, when its header says.
    total: int | None
    added: int
    duplicates: int
    error: str | None
    started_at: str
    finished_at: str | None


class Hit(Schema):
    chunk_id: int
    document_id: int
    document: str
    source: str
    text: str
    score: float


class Source(Hit):
    """A chunk added to a chat prompt; answers cite it as [index]."""

    index: int


class DataColumn(Schema):
    name: str
    type: str
    not_null: bool
    primary_key: bool
    about: str


class DataTable(Schema):
    name: str
    about: str
    # DELETE FROM it is allowed.
    deletable: bool
    columns: list[DataColumn]


class QueryExample(Schema):
    title: str
    sql: str


class DataSchema(Schema):
    tables: list[DataTable]
    examples: list[QueryExample]


class DeleteCount(Schema):
    table: str
    rows: int
    # Deleted (confirm), or only counted.
    done: bool


class QueryResult(Schema):
    columns: list[str]
    rows: list[list[Any]]
    # There were more rows than the limit.
    truncated: bool
    elapsed_ms: float
    # A DELETE: the rows it matches.
    delete: DeleteCount | None = None


# Web


class Website(Schema):
    """A saved crawl: start addresses and settings that can be run again."""

    id: int
    name: str
    collection: str
    urls: list[str]
    max_pages: int
    max_depth: int
    per_minute: float
    use_browser: bool
    respect_robots: bool
    only_under_start: bool
    exclude: list[str]
    created_at: str
    last_crawled_at: str | None


class CrawlJob(Schema):
    id: str
    collection: str
    options: CrawlOptions
    state: Literal["running", "done", "cancelled", "error"]
    website_id: int | None
    # New pages downloaded.
    fetched: int
    # Pages downloaded by earlier crawls, walked through without a request.
    known: int
    indexed: int
    duplicates: int
    skipped: int
    failed: int
    current: str | None
    # The last 50 events, newest last.
    log: list[str]
    # What happened to each start address (without #fragment); missing ones were not reached.
    outcomes: dict[str, Outcome]
    # Events so far, including those dropped from log.
    notes: int
    error: str | None
    started_at: str
    finished_at: str | None


class SearchResult(Schema):
    title: str
    url: str
    snippet: str
    kind: SearchKind
    thumbnail: str | None
    # The engine or site that found it.
    source: str
    site: str


class SearchOutcome(Schema):
    provider: Provider
    data: list[SearchResult]
    # Results dropped by the domain filter.
    hidden: int
    # The queries sent (with `site:` when the provider cannot filter by domain itself).
    queries: list[str]
    include_domains: list[str]
    exclude_domains: list[str]


class EngineSettingValues(Schema):
    # The proxy with its password as "****".
    proxy: str | None
    llama_cpp_dir: str | None
    # None: physical cores, at most 8.
    threads: int | None
    repetition_penalty: float
    rag_top_k: int
    embedding_device: Literal["auto", "cuda", "cpu", "mps"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class EngineSettingDefaults(EngineSettingValues):
    """The environment's values (TINYLM_*, HF_TOKEN), used for the settings not saved."""

    hf_token_hint: str | None


class PublicEngineSettings(EngineSettingValues):
    """The engine settings in use; the Hugging Face token is never sent back, only whether one
    is used, where it comes from and its first and last characters."""

    hf_token_set: bool
    hf_token_hint: str | None
    # app: saved in Settings; environment: HF_TOKEN or TINYLM_HF_TOKEN; login: `hf auth login`.
    hf_token_source: Literal["app", "environment", "login"] | None
    # The settings saved in the engine, which win over the environment's.
    saved: list[str]
    defaults: EngineSettingDefaults


class PublicSearchSettings(Schema):
    """The search settings; API keys are never sent back, only whether each is set."""

    provider: Provider
    include_domains: list[str]
    exclude_domains: list[str]
    searxng_url: str
    brave_api_key_set: bool
    tavily_api_key_set: bool
    tavily_depth: Literal["basic", "advanced"]
    tavily_topic: Literal["general", "news"]
    exa_api_key_set: bool
    exa_type: Literal["auto", "fast", "instant"]
    perplexity_api_key_set: bool
    serper_api_key_set: bool
    serper_country: str | None


# Datasets


class ExampleMessage(Schema):
    role: Literal["system", "user", "assistant"]
    content: str


class Dataset(Schema):
    name: str
    # Share of examples held out to measure the model on data it was not trained on.
    eval_percent: int
    created_at: str
    examples: int
    train: int
    eval: int


class Example(Schema):
    id: int
    messages: list[ExampleMessage]
    split: Literal["train", "eval"]
    origin: Origin
    # For generated examples: the document they were written from.
    source: str | None
    created_at: str


class AddedExample(Schema):
    # None for a duplicate.
    id: int | None
    duplicate: bool


class LineError(Schema):
    line: int
    error: str


class ImportResult(Schema):
    """A file of examples: how many were added, or why it could not be read."""

    name: str
    added: int | None = None
    duplicates: int | None = None
    invalid: int | None = None
    # The first bad records.
    errors: list[LineError] | None = None
    error: str | None = None


class TeacherCall(Schema):
    """One request to the teacher during a generation, and what came of it."""

    document: str
    # "write": the pairs from a chunk; "check": the pairs against the text.
    step: Literal["write", "check"]
    # As the API reported them (OpenRouter may route to another model or provider).
    model: str | None
    provider: str | None
    # None while the request runs.
    ok: bool | None
    result: str
    at: str


class GenerationJob(Schema):
    id: str
    dataset: str
    options: GenerationOptions
    state: Literal["running", "done", "cancelled", "error"]
    chunks_total: int
    chunks_done: int
    # Pairs the teacher wrote.
    generated: int
    added: int
    duplicates: int
    # Pairs dropped, by reason.
    rejected: dict[str, int]
    failed_chunks: int
    current: str | None
    # Seconds of a rate-limit pause in progress.
    waiting: float | None
    # The last 100 teacher requests, newest last.
    requests: list[TeacherCall]
    # The last 50 events.
    log: list[str]
    error: str | None
    created_at: str


class UnusedChunks(Schema):
    unused_chunks: int


# Teachers


class Teacher(Schema):
    """A model that writes training examples: a saved API provider or an installed model."""

    # None for an installed model.
    id: int | None
    # What generation and evaluation take: a saved teacher's id or "local:<model id>".
    ref: str
    name: str
    # The model id to chat with it: "tinylm/teacher/<name>", or a local teacher's model id.
    model_id: str
    kind: Literal["local", "openai", "anthropic", "cli"]
    base_url: str
    model: str
    has_key: bool
    per_minute: float | None
    max_tokens: int | None
    temperature: float | None
    thinking: Literal["default", "off", "on"]
    created_at: str | None


class TeacherModel(Schema):
    id: str
    name: str
    free: bool


class Served(Schema):
    """The model and provider that answered, as the API reported them."""

    model: str | None
    provider: str | None


class TeacherTest(Schema):
    reply: str
    served: Served | None


class CliLogin(Schema):
    state: Literal["running", "done", "failed", "cancelled"]
    # The tool's output, without terminal codes.
    output: str
    # The link to open in a browser.
    url: str | None
    # Waiting for the code the login page shows (Claude Code, Gemini CLI).
    wants_code: bool
    # A code was sent and the tool is checking it.
    checking_code: bool
    error: str | None


class CliTool(Schema):
    """A subscription's command-line tool (cli teachers) and its login."""

    tool: Literal["claude", "codex", "gemini"]
    name: str
    installed: bool
    version: str | None
    logged_in: bool
    # Who is logged in, when the tool says.
    account: str | None
    login: CliLogin | None


# Training, evaluation and export


class LossPoint(Schema):
    step: int
    loss: float | None = None
    eval_loss: float | None = None


class TrainingJob(Schema):
    id: str
    settings: TrainingSettings
    state: Literal["running", "stopping", "stopped", "done", "error", "interrupted"]
    created_at: str
    finished_at: str | None
    error: str | None
    pid: int | None
    step: int
    total_steps: int | None
    epoch: float
    loss: float | None
    eval_loss: float | None
    device: str | None
    train_examples: int | None
    eval_examples: int | None
    # Examples left out because their answer starts beyond max_length tokens.
    dropped: int
    seconds_per_step: float | None
    checkpoint_step: int | None
    resumable: bool
    eta_seconds: int | None
    # Only for a single run:
    history: list[LossPoint] | None = None
    # The worker's log, when the run failed.
    log: str | None = None


class EvaluationAnswer(Schema):
    text: str
    seconds: float
    # chrF against the reference, 0-100.
    similarity: float
    # The judge's score, 1-5.
    score: float | None
    reason: str | None


class EvaluationExample(Schema):
    messages: list[ExampleMessage]
    reference: str
    # By model id.
    answers: dict[str, EvaluationAnswer]
    judge_error: str | None = None


class EvaluationSummary(Schema):
    answers: int
    similarity: float | None
    judged: int
    score: float | None
    seconds: float | None
    chars: float | None


class Evaluation(Schema):
    id: str
    options: EvaluationOptions
    state: Literal["running", "done", "cancelled", "error", "interrupted"]
    created_at: str
    finished_at: str | None
    error: str | None
    phase: Literal["answering", "judging", ""]
    # The model answering.
    current: str | None
    done: int
    total: int
    # By model id.
    summary: dict[str, EvaluationSummary]
    # Only in a single report.
    examples: list[EvaluationExample] | None = None


class ExportJob(Schema):
    id: str
    model: str
    name: str
    quantization: Literal["Q8_0", "Q4_K_M", "F16"]
    state: Literal["running", "done", "error", "interrupted"]
    created_at: str
    finished_at: str | None
    error: str | None
    step: str | None
    steps: list[str]
    # Downloads and warnings.
    notes: list[str]
    # Bytes of the GGUF file.
    size: int | None


# OpenAI- and Anthropic-compatible endpoints


class Timings(Schema):
    """Per-answer figures (the names of llama.cpp server's `timings`)."""

    load_ms: float
    retrieval_ms: float
    prompt_n: int
    prompt_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_second: float
    total_ms: float


class OpenAIModel(Schema):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str


class OpenAIModelList(Schema):
    object: Literal["list"]
    data: list[OpenAIModel]


class AnthropicModel(Schema):
    type: Literal["model"]
    id: str
    display_name: str
    created_at: str


class AnthropicModelList(Schema):
    data: list[AnthropicModel]
    has_more: bool
    first_id: str | None
    last_id: str | None


class AssistantMessage(Schema):
    role: Literal["assistant"]
    content: str


class Choice(Schema):
    index: int
    message: AssistantMessage
    finish_reason: Literal["stop", "length"]


class Usage(Schema):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(Schema):
    """The answer when stream is false; with stream, the same fields come as
    chat.completion.chunk server-sent events."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    # tinylm extensions:
    timings: Timings
    # Only when a collection was used.
    sources: list[Source] | None = None
    collection: str | None = None
    # Only with debug: the request as the model got it, with the added sources.
    prompt: dict[str, Any] | None = None


class TextContent(Schema):
    type: Literal["text"]
    text: str


class MessageUsage(Schema):
    input_tokens: int
    output_tokens: int


class AnthropicMessage(Schema):
    """The answer when stream is false; with stream, Anthropic's server-sent events."""

    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[TextContent]
    stop_reason: Literal["end_turn", "stop_sequence", "max_tokens"] | None
    stop_sequence: str | None
    usage: MessageUsage
    # tinylm extensions:
    timings: Timings
    # Only when a collection was used.
    sources: list[Source] | None = None
    collection: str | None = None
