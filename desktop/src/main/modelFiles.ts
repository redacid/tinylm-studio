// Moves models between the engine and this computer: saves a model to a file (a fine-tuned
// model from a RunPod pod, say) and uploads a model file to the engine. Models are gigabytes, so
// this runs here, streaming to and from disk, rather than in the window's memory.
import { createWriteStream } from "node:fs";
import { open, rename, rm, stat } from "node:fs/promises";
import { basename } from "node:path";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { log } from "./log";

const logger = log("models");
// RunPod's proxy refuses bodies of 100 MB and requests longer than 100 s; 16 MB parts pass even
// over a slow upload.
const PART = 16 * 1024 * 1024;

export type Engine = { url: string; apiKey?: string };
export type Progress = { model: string; action: "save" | "upload"; done: number; total: number };

function headers(engine: Engine, extra: Record<string, string> = {}): Record<string, string> {
  return engine.apiKey ? { ...extra, authorization: `Bearer ${engine.apiKey}` } : extra;
}

async function failure(response: Response): Promise<Error> {
  let detail = response.statusText;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // not JSON: keep the status text
  }
  return new Error(`${detail} (HTTP ${response.status})`);
}

/** Saves the model's archive to path; a cancelled or failed download leaves no file behind. */
export async function saveModel(
  engine: Engine,
  model: string,
  path: string,
  progress: (p: Progress) => void,
): Promise<void> {
  logger.info(`Saving ${model} from ${engine.url} to ${path}`);
  const response = await fetch(`${engine.url}/api/models/${encodeURIComponent(model)}/archive`, {
    headers: headers(engine),
  });
  if (!response.ok || !response.body) throw await failure(response);
  const total = Number(response.headers.get("content-length") ?? 0);
  let done = 0;
  let reported = 0;
  const partial = `${path}.part`;
  const out = createWriteStream(partial);
  try {
    for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
      // A copy: the stream may reuse the memory of a chunk once it is handed over, while the
      // file is written later.
      if (!out.write(Buffer.from(chunk))) await once(out, "drain");
      done += chunk.length;
      if (done - reported >= 4 * 1024 * 1024 || done === total) {
        reported = done;
        progress({ model, action: "save", done, total });
      }
    }
    out.end();
    await finished(out);
    if (total && done !== total) throw new Error(`The download stopped at ${done} of ${total} bytes`);
    await rename(partial, path);
  } catch (error) {
    out.destroy();
    await rm(partial, { force: true });
    throw error;
  }
  logger.info(`Saved ${model} (${done} bytes)`);
}

/** Uploads a model archive (.tar) or a .gguf file in parts and has the engine install it;
 * returns the upload, which the window follows until the model is installed. */
export async function uploadModel(
  engine: Engine,
  path: string,
  progress: (p: Progress) => void,
): Promise<unknown> {
  const name = basename(path);
  const size = (await stat(path)).size;
  logger.info(`Uploading ${path} (${size} bytes) to ${engine.url}`);
  const json = async (response: Response) => {
    if (!response.ok) throw await failure(response);
    return (await response.json()) as { id: string };
  };
  const upload = await json(
    await fetch(`${engine.url}/api/models/uploads`, {
      method: "POST",
      headers: headers(engine, { "content-type": "application/json" }),
      body: JSON.stringify({ name, size }),
    }),
  );
  const file = await open(path, "r");
  try {
    const buffer = Buffer.alloc(PART);
    for (let offset = 0; offset < size; ) {
      const { bytesRead } = await file.read(buffer, 0, Math.min(PART, size - offset), offset);
      let tries = 0;
      // A dropped connection is retried a few times; the engine refuses a part twice.
      for (;;) {
        try {
          await json(
            await fetch(`${engine.url}/api/models/uploads/${upload.id}?offset=${offset}`, {
              method: "PUT",
              headers: headers(engine, { "content-type": "application/octet-stream" }),
              body: buffer.subarray(0, bytesRead),
            }),
          );
          break;
        } catch (error) {
          if (++tries >= 3 || !(error instanceof TypeError)) throw error;
          logger.warn(`Part at ${offset} failed (${error.message}); trying again`);
        }
      }
      offset += bytesRead;
      progress({ model: name, action: "upload", done: offset, total: size });
    }
  } finally {
    await file.close();
  }
  return json(
    await fetch(`${engine.url}/api/models/uploads/${upload.id}/finish`, { method: "POST", headers: headers(engine) }),
  );
}
