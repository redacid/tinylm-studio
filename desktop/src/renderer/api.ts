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

export class EngineClient {
  constructor(private readonly baseUrl: string) {}

  async installedModels(): Promise<string[]> {
    const body = await this.json<{ data: { id: string }[] }>("/v1/models");
    return body.data.map((m) => m.id);
  }

  async catalog(): Promise<CatalogEntry[]> {
    return (await this.json<{ catalog: CatalogEntry[] }>("/api/models")).catalog;
  }

  async download(modelId: string): Promise<void> {
    await this.json(`/api/models/${encodeURIComponent(modelId)}/download`, { method: "POST" });
  }

  /** Streams a chat completion, calling onText for every text fragment. */
  async chat(model: string, messages: ChatMessage[], onText: (text: string) => void, signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, messages, stream: true, max_tokens: 512 }),
      signal,
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
        const text = JSON.parse(data).choices?.[0]?.delta?.content;
        if (text) onText(text);
      }
    }
  }

  private async json<T = unknown>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, init);
    if (!response.ok) throw new Error(await errorText(response));
    return response.json() as Promise<T>;
  }
}

async function errorText(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.error?.message ?? body.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}
