// The app's own log: <user data>/logs/app.log, in the engine log's format ("date time.ms LEVEL
// scope: message"), so Settings › Logs shows both the same way. Kept to two files of 5 MB.
import { app } from "electron";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";

const MAX_BYTES = 5_000_000;
type Level = "DEBUG" | "INFO" | "WARNING" | "ERROR";

export function logFile(): string {
  return join(app.getPath("userData"), "logs", "app.log");
}

function stamp(date: Date): string {
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`
  );
}

function write(level: Level, scope: string, message: string) {
  // Polling (the pod's state every few seconds) is DEBUG: kept out unless asked for, as for the engine.
  if (level === "DEBUG" && process.env.TINYLM_LOG_LEVEL?.toUpperCase() !== "DEBUG") return;
  const line = `${stamp(new Date())} ${level} ${scope}: ${message}\n`;
  (level === "ERROR" || level === "WARNING" ? process.stderr : process.stdout).write(line);
  try {
    const path = logFile();
    mkdirSync(join(path, ".."), { recursive: true });
    if (existsSync(path) && statSync(path).size > MAX_BYTES) renameSync(path, `${path}.1`);
    appendFileSync(path, line);
  } catch {
    // Logging must never break the app.
  }
}

/** A logger for one part of the app: log("runpod").info("Pod created"). */
export function log(scope: string) {
  return {
    debug: (message: string) => write("DEBUG", scope, message),
    info: (message: string) => write("INFO", scope, message),
    warn: (message: string) => write("WARNING", scope, message),
    error: (message: string) => write("ERROR", scope, message),
  };
}

/** The last entries of the app's log, oldest first; following lines (a stack) join their entry. */
export function tail(lines: number): string[] {
  const path = logFile();
  if (!existsSync(path)) return [];
  const text = readFileSync(path, "utf8");
  const entries: string[] = [];
  for (const line of text.split("\n")) {
    if (!line) continue;
    if (/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d/.test(line) || entries.length === 0) entries.push(line);
    else entries[entries.length - 1] += `\n${line}`;
  }
  return entries.slice(-lines);
}
