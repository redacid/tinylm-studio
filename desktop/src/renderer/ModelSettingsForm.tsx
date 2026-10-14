import { useState } from "react";
import { EngineClient, InstalledModel, ModelSettings } from "./api";
import { Field, NumberInput } from "./Settings";

type NumberKey = Exclude<keyof ModelSettings, "thinking">;

type NumberField = {
  key: NumberKey;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  /** The value used when the field is empty. */
  empty: string;
};

const LOADING: NumberField[] = [
  {
    key: "context_size",
    label: "Context size",
    hint: "Tokens the model sees at once: the prompt, the sources from data and the answer. More takes more memory.",
    min: 256,
    max: 262144,
    step: 256,
    empty: "4096",
  },
  {
    key: "gpu_layers",
    label: "GPU layers",
    hint: "Layers put on the GPU when llama.cpp was built with GPU support; -1 is all, 0 runs on the CPU only.",
    min: -1,
    max: 1000,
    step: 1,
    empty: "-1",
  },
  {
    key: "batch_size",
    label: "Batch size",
    hint: "Prompt tokens read in one step. Larger reads long prompts faster but takes more memory.",
    min: 1,
    max: 8192,
    step: 64,
    empty: "512",
  },
];

const THREADS: NumberField = {
  key: "threads",
  label: "CPU threads",
  hint: "More than the physical cores, or than 8, rarely helps. Empty: the engine's setting (TINYLM_THREADS).",
  min: 1,
  max: 256,
  step: 1,
  empty: "auto",
};

const ANSWERS: NumberField[] = [
  {
    key: "max_tokens",
    label: "Max answer length",
    hint: "In tokens.",
    min: 1,
    max: 32768,
    step: 16,
    empty: "512",
  },
  {
    key: "temperature",
    label: "Temperature",
    hint: "0 always picks the most likely word. Higher is more varied.",
    min: 0,
    max: 2,
    step: 0.05,
    empty: "0.7",
  },
  {
    key: "top_p",
    label: "Top P",
    hint: "Picks only from the most likely words that together make up this share. 1 turns it off.",
    min: 0.01,
    max: 1,
    step: 0.05,
    empty: "1",
  },
  {
    key: "top_k",
    label: "Top K",
    hint: "Picks only from this many most likely words; 0 turns it off. Empty: the runtime's default, 40 in llama.cpp.",
    min: 0,
    max: 1000,
    step: 1,
    empty: "40",
  },
  {
    key: "min_p",
    label: "Min P",
    hint: "Drops words less likely than this share of the most likely one. Empty: the runtime's default, 0.05 in llama.cpp.",
    min: 0,
    max: 1,
    step: 0.01,
    empty: "0.05",
  },
  {
    key: "repetition_penalty",
    label: "Repetition penalty",
    hint: "Above 1 makes repeating words less likely; helps small models that loop. Empty: the engine's setting (TINYLM_REPETITION_PENALTY).",
    min: 0.5,
    max: 2,
    step: 0.05,
    empty: "1.1",
  },
];

/** How one installed model is loaded and the answer settings it uses unless a request sets them. */
export function ModelSettingsForm({
  client,
  model,
  onDone,
}: {
  client: EngineClient;
  model: InstalledModel;
  onDone: () => void;
}) {
  const [value, setValue] = useState<ModelSettings>(model.settings);
  const [error, setError] = useState<string | null>(null);
  const set = (key: keyof ModelSettings, v: number | boolean | null) => {
    const next = { ...value };
    if (v === null) delete next[key];
    else (next as Record<string, number | boolean>)[key] = v;
    setValue(next);
  };

  async function save(settings: ModelSettings) {
    setError(null);
    try {
      await client.saveModelSettings(model.id, settings);
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const numberField = (f: NumberField) => (
    <Field key={f.key} label={f.label} hint={f.hint}>
      <NumberInput
        value={value[f.key] ?? null}
        min={f.min}
        max={f.max}
        step={f.step}
        optional
        placeholder={f.empty}
        onChange={(n) => set(f.key, n)}
      />
    </Field>
  );

  return (
    <div className="model-settings settings-form">
      <h4>Loading</h4>
      <p className="muted">
        Used when the model is loaded; saving a change here unloads it, and the
        next request loads it again.
      </p>
      <div className="settings-grid">
        {model.format === "gguf" && LOADING.map(numberField)}
        {numberField(THREADS)}
      </div>
      <h4>Answers</h4>
      <p className="muted">
        Used when a request does not set them: by API clients, and by the chat
        for the fields left empty in Chat settings.
      </p>
      <div className="settings-grid">
        {ANSWERS.map(numberField)}
        <Field
          label="Thinking"
          hint="Models with a reasoning mode (Qwen3) think before they answer: better, but slower."
        >
          <select
            value={value.thinking ? "on" : ""}
            onChange={(e) => set("thinking", e.target.value ? true : null)}
          >
            <option value="">Off</option>
            <option value="on">On</option>
          </select>
        </Field>
      </div>
      <div className="buttons-row">
        <button onClick={() => save(value)}>Save</button>{" "}
        <button onClick={() => save({})}>Reset to defaults</button>{" "}
        <button onClick={onDone}>Cancel</button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
