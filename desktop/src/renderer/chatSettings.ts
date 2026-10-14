/** Chat settings, kept in the app's local storage so they survive restarts. */

export type ChatSettings = {
  /** The system prompt a new chat starts with. */
  systemPrompt: string;
  /** Empty (null): the model's setting from the Models tab, then the engine's default. */
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  /** Empty: the engine's default (TINYLM_REPETITION_PENALTY). */
  repetitionPenalty: number | null;
  /** Answering stops at any of these; empty lines are ignored. */
  stop: string[];
  /** How many chunks of the data an answer gets; empty: the engine's default (TINYLM_RAG_TOP_K). */
  sources: number | null;
  /** Show the answer as it is written; off, it appears when it is complete. */
  stream: boolean;
  /** The { } button on each message. */
  showJson: boolean;
  /** Start with the model and data chosen last time. */
  rememberChoice: boolean;
  /** Keep the conversation when the app is closed. */
  keepConversation: boolean;
  version?: number;
};

export const CHAT_DEFAULTS: ChatSettings = {
  systemPrompt: "",
  maxTokens: null,
  temperature: null,
  topP: null,
  repetitionPenalty: null,
  stop: [],
  sources: null,
  stream: true,
  showJson: true,
  rememberChoice: true,
  keepConversation: true,
};

const SETTINGS_KEY = "chat-settings";

export function loadChatSettings(): ChatSettings {
  const saved = read(SETTINGS_KEY) ?? {};
  if (!saved.version) {
    // Before version 2 the chat always sent these; the old defaults now mean "the model's".
    for (const [key, old] of [["maxTokens", 512], ["temperature", 0.7], ["topP", 1]] as const) {
      if (saved[key] === old) saved[key] = null;
    }
  }
  return { ...CHAT_DEFAULTS, ...saved, version: 2 };
}

export function saveChatSettings(settings: ChatSettings): void {
  write(SETTINGS_KEY, settings);
}

/** What the chat screen keeps between runs: the last choices and the conversation. */
export type ChatSession = {
  model?: string;
  collection?: string;
  system?: string;
  messages?: unknown[];
};

const SESSION_KEY = "chat-session";

export function loadChatSession(): ChatSession {
  return read(SESSION_KEY) ?? {};
}

/** False when the storage is full. */
export function saveChatSession(session: ChatSession): boolean {
  return write(SESSION_KEY, session);
}

function read(key: string) {
  try {
    return JSON.parse(localStorage.getItem(key) ?? "null");
  } catch {
    return null;
  }
}

function write(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false; // full or unavailable storage: kept until the app is closed
  }
}
