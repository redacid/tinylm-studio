import { FormEvent, useCallback, useEffect, useState } from "react";
import { EngineClient, Teacher, TeacherInput, TeacherModel, ThinkingMode } from "./api";
import { servedBy } from "./Generate";
import { useTabActive } from "./tabActive";
import { Icon } from "./Icon";
import { BusySpinner, useBusy } from "./Busy";

/** Ready-made settings for common providers; every field stays editable. */
const PROVIDERS = [
  { label: "OpenRouter", kind: "openai", base_url: "https://openrouter.ai/api/v1", per_minute: 20 },
  { label: "OpenAI", kind: "openai", base_url: "https://api.openai.com/v1", per_minute: null },
  { label: "Groq", kind: "openai", base_url: "https://api.groq.com/openai/v1", per_minute: 30 },
  { label: "Anthropic", kind: "anthropic", base_url: "https://api.anthropic.com", per_minute: null },
  { label: "Ollama (this computer)", kind: "openai", base_url: "http://localhost:11434/v1", per_minute: null },
  { label: "LM Studio (this computer)", kind: "openai", base_url: "http://localhost:1234/v1", per_minute: null },
  // Subscriptions: the engine runs the tool, logged in once under Settings › Subscriptions.
  { label: "Claude subscription (Claude Code)", kind: "cli", base_url: "claude", per_minute: null },
  { label: "ChatGPT subscription (Codex CLI)", kind: "cli", base_url: "codex", per_minute: null },
  { label: "Google account (Gemini CLI)", kind: "cli", base_url: "gemini", per_minute: null },
] as const;


/** A model each tool takes, for the Model field's placeholder. */
const CLI_EXAMPLES: Record<string, string> = { claude: "sonnet", codex: "gpt-5.5", gemini: "flash" };

/** Which tool a program is ("claude" for /usr/local/bin/claude). */
function cliTool(program: string | undefined): string {
  const name = (program || "claude").split("/").pop() ?? "";
  return Object.keys(CLI_EXAMPLES).find((tool) => name.startsWith(tool)) ?? "";
}

const EMPTY: TeacherInput = {
  id: null,
  name: "",
  kind: "openai",
  base_url: PROVIDERS[0].base_url,
  model: "",
  api_key: "",
  per_minute: PROVIDERS[0].per_minute,
  max_tokens: null,
  temperature: null,
  thinking: "default",
};

const number = (value: string) => (value ? +value : null);

/** Teachers: the larger models that write training examples. */
export function Teachers({ client }: { client: EngineClient }) {
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [editing, setEditing] = useState<TeacherInput | null>(null);
  const [hasKey, setHasKey] = useState(false);
  const [models, setModels] = useState<TeacherModel[]>([]);
  const [freeOnly, setFreeOnly] = useState(true);
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  const loading = useBusy();
  const { run: load } = loading;
  const refresh = useCallback(() => load("Loading teachers…", () => client.teachers()).then(setTeachers, report), [client, load]);

  const active = useTabActive();
  useEffect(() => {
    if (active) refresh();
  }, [refresh, active]);

  function edit(teacher: Teacher | null) {
    setError(null);
    setResult(null);
    setModels([]);
    setHasKey(teacher?.has_key ?? false);
    setEditing(
      teacher && teacher.kind !== "local"
        ? { ...teacher, kind: teacher.kind, api_key: "" }
        : { ...EMPTY },
    );
  }

  const set = (patch: Partial<TeacherInput>) => setEditing((t) => (t ? { ...t, ...patch } : t));

  function pickProvider(label: string) {
    const provider = PROVIDERS.find((p) => p.label === label);
    if (provider) set({ kind: provider.kind, base_url: provider.base_url, per_minute: provider.per_minute });
    setModels([]);
  }

  async function check(action: () => Promise<void>) {
    setChecking(true);
    setResult(null);
    try {
      await action();
    } catch (e) {
      setResult({ ok: false, text: e instanceof Error ? e.message : String(e) });
    }
    setChecking(false);
  }

  const listModels = () => check(async () => setModels(await client.teacherModels(editing!)));
  const test = () =>
    check(async () => {
      const { reply, served } = await client.testTeacher(editing!);
      setResult({ ok: true, text: `The model answered: ${reply}${served ? ` (${servedBy(served)})` : ""}` });
    });

  async function save(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      await client.saveTeacher(editing!);
      setEditing(null);
      await refresh();
    } catch (e) {
      report(e);
    }
  }

  async function remove(id: number) {
    setConfirmDelete(null);
    await client.deleteTeacher(id).catch(report);
    await refresh();
  }

  const saved = teachers.filter((t) => t.kind !== "local");
  const local = teachers.filter((t) => t.kind === "local");
  const provider = PROVIDERS.find((p) => p.base_url === editing?.base_url && p.kind === editing?.kind)?.label ?? "";
  const cli = editing?.kind === "cli";
  const ready = cli || !!editing?.model?.trim();
  const shown = freeOnly && models.some((m) => m.free) ? models.filter((m) => m.free) : models;

  return (
    <div className="teachers">
      <div className="websites-head">
        <h3>Teachers</h3>
        <button onClick={() => edit(null)} disabled={editing !== null}><Icon name="plus" />Add teacher</button>
      </div>
      <p className="muted">
        Teachers are larger models that write training examples from your documents (Datasets → Generate). Use any
        OpenAI-compatible API, such as OpenRouter with its free models, or Anthropic, or a subscription through its
        command-line tool (Claude Code, Codex, Gemini CLI). Installed models can be teachers too
        {local.length > 0 ? `: ${local.map((t) => t.name).join(", ")}` : ""}; they are free and private but slow on a CPU.
      </p>
      {error && <p className="error">{error}</p>}
      <BusySpinner busy={loading} block />
      {editing && (
        <form className="teacher-form" onSubmit={save}>
          <label>
            <span>Provider</span>
            <select value={provider} onChange={(e) => pickProvider(e.target.value)}>
              {PROVIDERS.map((p) => (
                <option key={p.label} value={p.label}>{p.label}</option>
              ))}
              <option value="">Other</option>
            </select>
          </label>
          <label>
            <span>API</span>
            <select value={editing.kind} onChange={(e) => set({ kind: e.target.value as TeacherInput["kind"] })}>
              <option value="openai">OpenAI-compatible</option>
              <option value="anthropic">Anthropic</option>
              <option value="cli">Command-line tool (subscription)</option>
            </select>
          </label>
          <label title={cli ? "claude, codex or gemini, or the full path to one" : undefined}>
            <span>{cli ? "Program" : "Base URL"}</span>
            <input value={editing.base_url ?? ""} onChange={(e) => set({ base_url: e.target.value })} />
          </label>
          {cli && (
            <p className="muted">
              The engine runs <code>{editing.base_url || "claude"}</code> on its computer for every request, without its
              tools. Log the tool in once under <strong>Settings › Subscriptions</strong>; the answers then count toward
              that subscription's limits.
            </p>
          )}
          {!cli && <label>
            <span>API key</span>
            <input
              type="password"
              autoComplete="off"
              placeholder={hasKey ? "Saved; leave empty to keep it" : "Not needed for a server on this computer"}
              value={editing.api_key ?? ""}
              onChange={(e) => set({ api_key: e.target.value })}
            />
          </label>}
          <label>
            <span>Model</span>
            <input
              list="teacher-models"
              placeholder={cli ? `Empty: the tool's default, e.g. ${CLI_EXAMPLES[cliTool(editing.base_url)] ?? "sonnet"}` : "e.g. meta-llama/llama-3.3-70b-instruct:free"}
              value={editing.model}
              onChange={(e) => set({ model: e.target.value })}
            />
            <datalist id="teacher-models">
              {shown.map((m) => (
                <option key={m.id} value={m.id}>{m.free ? `${m.name} (free)` : m.name}</option>
              ))}
            </datalist>
            <button type="button" onClick={listModels} disabled={checking}><Icon name="list" />List models</button>
          </label>
          {models.length > 0 && (
            <label>
              <span />
              {cli ? (
                <span className="muted">
                  {models.length} models: {models.map((m) => m.id).join(", ")}. Pick one in the Model field.
                </span>
              ) : (
                <span className="muted">
                  {models.length} models, {models.filter((m) => m.free).length} free.{" "}
                  <input type="checkbox" checked={freeOnly} onChange={(e) => setFreeOnly(e.target.checked)} /> Free only
                </span>
              )}
            </label>
          )}
          <label>
            <span>Name</span>
            <input placeholder={cli ? "Defaults to the model or the tool" : "Defaults to the model"} value={editing.name ?? ""} onChange={(e) => set({ name: e.target.value })} />
          </label>
          <label title="Free plans limit requests per minute; the studio waits between requests">
            <span>Requests per minute</span>
            <input
              type="number"
              min={1}
              step="any"
              placeholder="No limit"
              value={editing.per_minute ?? ""}
              onChange={(e) => set({ per_minute: number(e.target.value) })}
            />
          </label>
          {!cli && <>
          <p className="muted">
            Answer settings apply to everything this teacher does: the chat, generating examples, judging. Left empty,
            each request uses its own.
          </p>
          <label title="Tokens per answer. Thinking counts toward them: a model that thinks needs a few thousand">
            <span>Max tokens</span>
            <input
              type="number"
              min={1}
              placeholder="Each request's own"
              value={editing.max_tokens ?? ""}
              onChange={(e) => set({ max_tokens: number(e.target.value) })}
            />
          </label>
          <label title="Some models, such as newer Claude models, take none; the studio then asks without one">
            <span>Temperature</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              placeholder="Each request's own"
              value={editing.temperature ?? ""}
              onChange={(e) => set({ temperature: number(e.target.value) })}
            />
          </label>
          <label title="Thinking before answering, for models that have it (Claude 4.6 and later; OpenRouter's reasoning models)">
            <span>Thinking</span>
            <select
              value={editing.thinking ?? "default"}
              onChange={(e) => set({ thinking: e.target.value as ThinkingMode })}
            >
              <option value="default">The model's default</option>
              <option value="off">Off</option>
              <option value="on">On</option>
            </select>
          </label>
          </>}
          {result && <p className={result.ok ? "muted" : "error"}>{result.text}</p>}
          <div className="buttons">
            <button type="button" onClick={test} disabled={checking || !ready}>
              <Icon name={checking ? "loader-2" : "flask"} spin={checking} />
              {checking ? "Checking…" : "Test"}
            </button>
            <button type="button" onClick={() => setEditing(null)}><Icon name="x" />Cancel</button>
            <button type="submit" disabled={!ready}><Icon name="device-floppy" />Save</button>
          </div>
        </form>
      )}
      {saved.length > 0 && (
        <table>
          <thead>
            <tr><th>Teacher</th><th>Provider</th><th>Limit</th><th>Answers</th><th /></tr>
          </thead>
          <tbody>
            {saved.map((t) => (
              <tr key={t.ref}>
                <td>
                  {t.name}
                  {t.model && t.name !== t.model && <div className="muted">{t.model}</div>}
                </td>
                <td>
                  {t.base_url}
                  <div className="muted">
                    {t.kind === "cli"
                      ? "Command-line tool · subscription"
                      : `${t.kind === "anthropic" ? "Anthropic API" : "OpenAI-compatible"} · ${t.has_key ? "key saved" : "no key"}`}
                  </div>
                </td>
                <td className="nowrap">{t.per_minute ? `${t.per_minute} / min` : "none"}</td>
                <td className="nowrap">{t.kind === "cli" ? "the tool's" : answerSettings(t)}</td>
                <td className="nowrap">
                  {confirmDelete === t.id ? (
                    <>
                      <button className="danger" onClick={() => remove(t.id!)}><Icon name="trash" />Delete</button>{" "}
                      <button onClick={() => setConfirmDelete(null)}><Icon name="x" />Cancel</button>
                    </>
                  ) : (
                    <>
                      <button onClick={() => edit(t)} disabled={editing !== null}><Icon name="pencil" />Edit</button>{" "}
                      <button onClick={() => setConfirmDelete(t.id)}><Icon name="trash" />Delete</button>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted">API keys are kept in the workspace database on this computer and never shown again.</p>
    </div>
  );
}

function answerSettings(t: Teacher): string {
  const parts = [
    t.max_tokens ? `${t.max_tokens} tokens` : "",
    t.temperature !== null ? `temperature ${t.temperature}` : "",
    t.thinking !== "default" ? `thinking ${t.thinking}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "defaults";
}
