#!/usr/bin/env node
// Records a GIF walkthrough of the desktop app from a YAML scenario (see docs/demo.md).
//
// Playwright drives the real Electron app, whose engine works in a data folder of its own
// (<data dir>/demo), so the walkthrough starts empty and leaves the user's data alone. Each action
// is captured as frames; long waits (a crawl, training) become a fast-forwarded timelapse. The
// captions of every language are drawn on the same frames afterwards, and ffmpeg turns each set
// into a GIF.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");
const desktop = path.join(repo, "desktop");
// The recorder uses the desktop app's packages: Electron itself, playwright-core and yaml.
const require = createRequire(path.join(desktop, "package.json"));
const { _electron } = require("playwright-core");
const YAML = require("yaml");
const electronPath = require("electron");

const USAGE = `Usage: node scripts/demo/record.mjs [options]
  --scenario FILE    the scenario (default scripts/demo/scenario.yaml)
  --out DIR          where the GIFs go (default docs/images)
  --name NAME        GIF file name: NAME-<lang>.gif (default: the scenario's name, else walkthrough)
  --lang LIST        caption languages, comma-separated (default: all in the scenario)
  --width PX         GIF width (default 960)
  --data-dir DIR     the engine's data folder whose models are used (default as the app's)
  --var NAME=VALUE   override a scenario variable (repeatable)
  --keep             keep the demo's data folder from the previous run instead of starting empty`;

// Timing of the GIF, in milliseconds.
const CURSOR_FRAMES = 7;
const CURSOR_FRAME_MS = 45;
const TYPE_FRAMES = 8;
const TYPE_FRAME_MS = 70;
const AFTER_ACTION_MS = 500;
const POINT_MS = 1200;
const TIMELAPSE_FRAME_MS = 150;
const TIMELAPSE_MAX_FRAMES = 40;

function parseArgs(argv) {
  const opts = { vars: {}, keep: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) fail(`${arg} needs a value`);
      return argv[++i];
    };
    if (arg === "--scenario") opts.scenario = value();
    else if (arg === "--out") opts.out = value();
    else if (arg === "--name") opts.name = value();
    else if (arg === "--lang") opts.lang = value();
    else if (arg === "--width") opts.width = Number(value());
    else if (arg === "--data-dir") opts.dataDir = value();
    else if (arg === "--keep") opts.keep = true;
    else if (arg === "--var") {
      const [name, ...rest] = value().split("=");
      opts.vars[name] = rest.join("=");
    } else if (arg === "--help" || arg === "-h") {
      console.log(USAGE);
      process.exit(0);
    } else fail(`Unknown option: ${arg}\n${USAGE}`);
  }
  return opts;
}

function fail(message) {
  console.error(message);
  process.exit(2);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Replaces ${name} in every string of a value with the variable's value. */
function interpolate(value, vars) {
  if (typeof value === "string")
    return value.replace(/\$\{(\w+)\}/g, (_, name) => {
      if (!(name in vars)) throw new Error(`Unknown variable \${${name}}`);
      return String(vars[name]);
    });
  if (Array.isArray(value)) return value.map((v) => interpolate(v, vars));
  if (value && typeof value === "object")
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, interpolate(v, vars)]));
  return value;
}

// ---- The demo's data folder -------------------------------------------------------------------

function defaultDataDir() {
  const base = process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share");
  return path.join(base, "tinylm-studio");
}

function installedModels(modelsDir) {
  if (!fs.existsSync(modelsDir)) return [];
  return fs
    .readdirSync(modelsDir, { withFileTypes: true })
    .filter((e) => (e.isDirectory() || e.isSymbolicLink()) && !e.name.startsWith("."))
    .map((e) => e.name);
}

function folderSize(dir) {
  return fs.readdirSync(dir).reduce((sum, name) => sum + fs.statSync(path.join(dir, name)).size, 0);
}

/** A GGUF model that runs on its own (not an adapter of another model). */
function isGguf(dir) {
  return fs.readdirSync(dir).some((f) => f.endsWith(".gguf")) && !fs.existsSync(path.join(dir, "tinylm.json"));
}

/**
 * Sets the teacher variables: teacher (empty when examples are imported instead) and
 * teacher_name, as the Generate form lists it. Returns the API teacher to add, if any.
 */
function resolveTeacher(vars, modelsDir) {
  const { TEACHER_URL: url, TEACHER_MODEL: model, TEACHER_KEY: key } = process.env;
  if (url || model) {
    if (!model) throw new Error("TEACHER_URL needs TEACHER_MODEL too");
    vars.teacher = model;
    vars.teacher_name = model;
    return { name: model, kind: "openai", base_url: url ?? "", model, api_key: key ?? null };
  }
  const installed = installedModels(modelsDir);
  if (vars.teacher === "auto") {
    const gguf = installed.filter((m) => isGguf(path.join(modelsDir, m)));
    gguf.sort((a, b) => folderSize(path.join(modelsDir, a)) - folderSize(path.join(modelsDir, b)));
    vars.teacher = gguf[0] ?? "";
  } else if (vars.teacher && !installed.includes(vars.teacher)) {
    throw new Error(`The teacher ${vars.teacher} is not installed (make download MODEL=${vars.teacher})`);
  }
  vars.teacher_name = vars.teacher;
  return null;
}

/** An empty data folder for the demo, with the models it needs linked from the real one. */
function prepareDataDir(demoDir, realDir, models, keep) {
  if (!keep) fs.rmSync(demoDir, { recursive: true, force: true });
  fs.mkdirSync(path.join(demoDir, "models"), { recursive: true });
  for (const model of models) {
    const source = path.join(realDir, "models", model);
    if (!fs.existsSync(source))
      throw new Error(`The model ${model} is not installed in ${realDir} (make download MODEL=${model})`);
    const link = path.join(demoDir, "models", model);
    if (!fs.existsSync(link)) fs.symlinkSync(source, link);
  }
  // The embedding model is shared, so it is downloaded once, not on every run.
  const embeddings = path.join(realDir, "embeddings");
  fs.mkdirSync(embeddings, { recursive: true });
  const link = path.join(demoDir, "embeddings");
  if (!fs.existsSync(link)) fs.symlinkSync(embeddings, link);
}

// ---- Driving the app and capturing frames ------------------------------------------------------

const CURSOR_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">' +
  '<path d="M4 2l16 10.5-7 1.2 4.2 7.6-3 1.6-4.2-7.7L4 20z" fill="#111" stroke="#fff" stroke-width="1.5"/></svg>';

class Recorder {
  constructor(page, frameDir) {
    this.page = page;
    this.frameDir = frameDir;
    this.frames = []; // { file, ms, step, speed? }
    this.step = 0;
    this.count = 0;
    this.cursor = { x: 40, y: 40 };
    // Off for steps that only prepare something (record: false): they leave no frames.
    this.recording = true;
  }

  /** The fake mouse pointer: Playwright's clicks do not move the real one. */
  async injectCursor() {
    await this.page.evaluate(
      ({ svg, x, y }) => {
        if (document.getElementById("demo-cursor")) return;
        const style = document.createElement("style");
        style.textContent =
          "#demo-cursor{position:fixed;z-index:2147483647;pointer-events:none;width:22px;height:22px}" +
          "#demo-cursor.click::before{content:'';position:absolute;left:-11px;top:-11px;width:22px;height:22px;" +
          "border-radius:50%;background:rgba(250,180,40,.55)}" +
          "#demo-highlight{position:fixed;z-index:2147483646;pointer-events:none;display:none;border-radius:8px;" +
          "border:3px solid rgb(245,158,11);box-shadow:0 0 0 5px rgba(245,158,11,.25)}";
        document.head.append(style);
        const cursor = document.createElement("div");
        cursor.id = "demo-cursor";
        cursor.innerHTML = svg;
        cursor.style.left = `${x}px`;
        cursor.style.top = `${y}px`;
        const highlight = document.createElement("div");
        highlight.id = "demo-highlight";
        document.body.append(cursor, highlight);
      },
      { svg: CURSOR_SVG, ...this.cursor },
    );
  }

  async shot(ms, extra = {}) {
    if (!this.recording) return { file: null, ms, step: this.step, ...extra };
    const file = path.join(this.frameDir, `raw-${String(++this.count).padStart(5, "0")}.png`);
    await this.page.screenshot({ path: file });
    const frame = { file, ms, step: this.step, ...extra };
    this.frames.push(frame);
    return frame;
  }

  async hold(ms) {
    if (!this.recording) return;
    if (this.frames.length === 0) await this.shot(0);
    this.frames[this.frames.length - 1].ms += ms;
  }

  locate(spec) {
    if (!spec || typeof spec !== "object") throw new Error(`Not a target: ${JSON.stringify(spec)}`);
    const page = this.page;
    const scope = spec.within ? page.locator(spec.within).filter({ visible: true }).first() : page;
    let locator;
    if (spec.tab) locator = page.locator("header nav").getByRole("button", { name: spec.tab, exact: true });
    else if (spec.button) locator = scope.getByRole("button", { name: spec.button, exact: true });
    else if (spec.placeholder) locator = scope.getByPlaceholder(spec.placeholder, { exact: true });
    else if (spec.label) locator = scope.getByLabel(spec.label, { exact: !!spec.exact });
    else if (spec.text) locator = scope.getByText(spec.text, { exact: !!spec.exact });
    else if (spec.css) locator = scope.locator(spec.css);
    else throw new Error(`A target needs tab, button, placeholder, label, text or css: ${JSON.stringify(spec)}`);
    // Hidden tabs stay mounted, so only what is on screen counts (file inputs are always hidden).
    if (!spec.hidden) locator = locator.filter({ visible: true });
    return locator.nth(spec.nth ?? 0);
  }

  /** Glides the pointer to the middle of an element, capturing the way there. */
  async moveTo(locator, click = true) {
    await locator.waitFor({ state: "visible", timeout: 30_000 });
    await locator.scrollIntoViewIfNeeded();
    const box = await locator.boundingBox();
    if (!box) throw new Error("The element has no position on screen");
    const to = { x: box.x + Math.min(box.width / 2, 60), y: box.y + box.height / 2 };
    const from = this.cursor;
    for (let i = 1; i <= CURSOR_FRAMES; i++) {
      const t = i / CURSOR_FRAMES;
      const ease = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      await this.placeCursor(from.x + (to.x - from.x) * ease, from.y + (to.y - from.y) * ease, false);
      await this.shot(CURSOR_FRAME_MS);
    }
    this.cursor = to;
    if (click) {
      await this.placeCursor(to.x, to.y, true);
      await this.shot(120);
      await this.placeCursor(to.x, to.y, false);
    }
  }

  async placeCursor(x, y, clicking) {
    await this.page.evaluate(
      ({ x, y, clicking }) => {
        const cursor = document.getElementById("demo-cursor");
        if (!cursor) return;
        cursor.style.left = `${x}px`;
        cursor.style.top = `${y}px`;
        cursor.classList.toggle("click", clicking);
      },
      { x, y, clicking },
    );
  }

  /** Draws a frame around a part of the screen (box: null removes it). */
  async highlight(box) {
    await this.page.evaluate((box) => {
      const highlight = document.getElementById("demo-highlight");
      if (!highlight) return;
      highlight.style.display = box ? "block" : "none";
      if (!box) return;
      const pad = 5;
      highlight.style.left = `${box.x - pad}px`;
      highlight.style.top = `${box.y - pad}px`;
      highlight.style.width = `${box.width + 2 * pad}px`;
      highlight.style.height = `${box.height + 2 * pad}px`;
    }, box);
  }

  /** Points at a part of the screen and frames it, without clicking: for explaining it. */
  async point({ hold = 0, ...spec }) {
    const locator = this.locate(spec);
    await this.moveTo(locator, false);
    await this.highlight(await locator.boundingBox());
    await this.shot(POINT_MS + Number(hold) * 1000);
  }

  async click(spec) {
    const locator = this.locate(spec);
    await this.moveTo(locator);
    await locator.click();
    // The real (invisible) mouse would leave hover highlights wherever it rests.
    await this.page.mouse.move(0, 0);
    await sleep(250);
    await this.shot(AFTER_ACTION_MS);
  }

  async fill({ text, secret = false, ...spec }) {
    const locator = this.locate(spec);
    await this.moveTo(locator);
    // A key typed into a GIF would be published with it: show dots instead.
    if (secret) await locator.evaluate((el) => (el.type = "password"));
    // Typing over the selected text: a cleared number field would show 0 in front of the digits.
    await locator.click();
    await locator.press("ControlOrMeta+a");
    const value = String(text);
    const size = Math.max(1, Math.ceil(value.length / TYPE_FRAMES));
    for (let i = 0; i < value.length; i += size) {
      await locator.pressSequentially(value.slice(i, i + size));
      await this.shot(TYPE_FRAME_MS);
    }
    await this.hold(AFTER_ACTION_MS - TYPE_FRAME_MS);
  }

  async check(spec) {
    const locator = this.locate(spec);
    await this.moveTo(locator);
    await locator.check();
    await this.shot(AFTER_ACTION_MS);
  }

  /** Picks an option by its text (partial: one that contains it), once the list has it. */
  async select({ option, partial, ...spec }) {
    const locator = this.locate(spec);
    await this.moveTo(locator);
    const deadline = Date.now() + 30_000;
    for (;;) {
      const labels = (await locator.locator("option").allTextContents()).map((t) => t.trim());
      const label = labels.find((t) => (partial ? t.includes(option) : t === option));
      if (label) {
        await locator.selectOption({ label });
        break;
      }
      if (Date.now() > deadline) throw new Error(`No option "${option}" in ${JSON.stringify(labels)}`);
      await sleep(500);
    }
    await this.shot(AFTER_ACTION_MS);
  }

  async upload({ file, ...spec }, baseDir) {
    const input = this.locate({ ...spec, hidden: true });
    // The input is hidden inside a label that looks like a button: point at the label.
    await this.moveTo(input.locator("xpath=.."));
    await input.setInputFiles(path.resolve(baseDir, file));
    await sleep(250);
    await this.shot(AFTER_ACTION_MS);
  }

  async scroll({ block = "start", ...spec }) {
    const locator = this.locate(spec);
    await locator.waitFor({ state: "visible", timeout: 30_000 });
    await locator.evaluate((el, block) => el.scrollIntoView({ block }), block);
    await sleep(200);
    await this.shot(AFTER_ACTION_MS);
  }

  /**
   * Waits until a target shows up (for) or goes away (gone). With timelapse, the screen is
   * captured all along and played back faster: at most TIMELAPSE_MAX_FRAMES frames, so an hour
   * of training takes a few seconds of the GIF.
   */
  async wait({ for: shown, gone, timeout = 60, timelapse = false, frames = TIMELAPSE_MAX_FRAMES }) {
    timelapse &&= this.recording;
    if (!shown && !gone) throw new Error("wait needs for or gone");
    const locator = this.locate(shown ?? gone);
    // What should go away may not have appeared yet.
    if (gone) await locator.waitFor({ state: "visible", timeout: 5000 }).catch(() => {});
    const done = async () => (shown ? locator.isVisible() : !(await locator.isVisible()));
    const start = Date.now();
    if (!timelapse) {
      await locator.waitFor({ state: shown ? "visible" : "hidden", timeout: timeout * 1000 });
      await this.shot(AFTER_ACTION_MS);
      return;
    }
    const snaps = [];
    let interval = 1000;
    for (;;) {
      const finished = await done();
      snaps.push(await this.shot(0));
      if (finished) break;
      if (Date.now() - start > timeout * 1000) throw new Error(`Still waiting after ${timeout} s`);
      await sleep(interval);
      if (snaps.length >= 2 * frames) {
        // Keeps every other frame (always the first) and captures half as often from now on.
        const dropped = snaps.filter((_, i) => i % 2 === 1);
        for (const frame of dropped) fs.rmSync(frame.file);
        this.frames = this.frames.filter((f) => !dropped.includes(f));
        snaps.splice(0, snaps.length, ...snaps.filter((_, i) => i % 2 === 0));
        interval *= 2;
      }
    }
    const elapsed = Date.now() - start;
    // A wait of a few seconds is shown as it was.
    const speed = elapsed > 5000 ? Math.round(elapsed / (snaps.length * TIMELAPSE_FRAME_MS)) : 1;
    for (const frame of snaps) {
      frame.ms = speed >= 2 ? TIMELAPSE_FRAME_MS : interval;
      if (speed >= 2) frame.speed = speed;
    }
    // The result, at normal speed.
    snaps[snaps.length - 1].speed = undefined;
    await this.hold(AFTER_ACTION_MS);
    console.log(`  waited ${Math.round(elapsed / 1000)} s${speed >= 2 ? `, shown ${speed}× faster` : ""}`);
  }

  /** Opens an address in the app's window, e.g. the engine's Swagger UI. */
  async open(url) {
    await this.page.goto(url);
    await this.page.waitForLoadState("networkidle").catch(() => {});
    await this.injectCursor();
    await this.shot(AFTER_ACTION_MS);
  }

  /** Adds CSS to the page, e.g. to hide what the scenario is not about. */
  async style(css) {
    await this.page.addStyleTag({ content: css });
    await sleep(200);
  }

  async run(action, baseDir) {
    const [kind, arg] = Object.entries(action)[0];
    // A frame drawn by point stays until the next action.
    if (kind !== "pause") await this.highlight(null);
    switch (kind) {
      case "point":
        return this.point(arg);
      case "click":
        return this.click(arg);
      case "fill":
        return this.fill(arg);
      case "check":
        return this.check(arg);
      case "select":
        return this.select(arg);
      case "upload":
        return this.upload(arg, baseDir);
      case "scroll":
        return this.scroll(arg);
      case "wait":
        return this.wait(arg);
      case "style":
        return this.style(arg);
      case "open":
        return this.open(arg);
      case "pause":
        return this.hold(Number(arg) * 1000);
      default:
        throw new Error(`Unknown action: ${kind}`);
    }
  }
}

// ---- Captions and the GIF ----------------------------------------------------------------------

/** How long a caption must stay on screen to be read: about 15 characters a second. */
const readingMs = (caption) => Math.max(2500, ...Object.values(caption).map((text) => 600 + text.length * 65));

/**
 * Draws a frame with its caption below it (and the fast-forward badge), in a hidden window: the
 * app's own page does not allow data: images. scale is the screen's device pixel ratio.
 */
async function compose(page, frame, caption, badge, scale) {
  const png = fs.readFileSync(frame.file).toString("base64");
  const url = await page.evaluate(
    async ({ png, caption, badge, scale }) => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const band = Math.round(70 * scale);
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height + band;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);
      ctx.fillStyle = "#14161c";
      ctx.fillRect(0, image.height, canvas.width, band);
      const family = 'system-ui, "Noto Sans", "DejaVu Sans", sans-serif';
      // One line when it fits, otherwise two at a smaller size.
      const pad = 24 * scale;
      let size = 22;
      let lines = [caption];
      ctx.font = `600 ${size * scale}px ${family}`;
      if (ctx.measureText(caption).width > canvas.width - 2 * pad) {
        size = 18;
        ctx.font = `600 ${size * scale}px ${family}`;
        const words = caption.split(" ");
        let best = 1;
        for (let i = 1; i < words.length; i++) {
          const first = ctx.measureText(words.slice(0, i).join(" ")).width;
          const second = ctx.measureText(words.slice(i).join(" ")).width;
          const current = Math.max(
            ctx.measureText(words.slice(0, best).join(" ")).width,
            ctx.measureText(words.slice(best).join(" ")).width,
          );
          if (Math.max(first, second) < current) best = i;
        }
        lines = [words.slice(0, best).join(" "), words.slice(best).join(" ")];
      }
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const lineHeight = size * 1.35 * scale;
      lines.forEach((line, i) => {
        const y = image.height + band / 2 + (i - (lines.length - 1) / 2) * lineHeight;
        ctx.fillText(line, canvas.width / 2, y);
      });
      if (badge) {
        ctx.font = `700 ${16 * scale}px ${family}`;
        const width = ctx.measureText(badge).width + 44 * scale;
        const height = 32 * scale;
        const x = canvas.width - width - 16 * scale;
        const y = 56 * scale;
        ctx.fillStyle = "rgba(217, 119, 6, 0.92)";
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, height / 2);
        ctx.fill();
        // Two triangles: the fast-forward sign (emoji fonts are not everywhere).
        ctx.fillStyle = "#ffffff";
        const t = 7 * scale;
        for (const dx of [0, t]) {
          ctx.beginPath();
          ctx.moveTo(x + 12 * scale + dx, y + height / 2 - t);
          ctx.lineTo(x + 12 * scale + dx + t, y + height / 2);
          ctx.lineTo(x + 12 * scale + dx, y + height / 2 + t);
          ctx.fill();
        }
        ctx.textAlign = "left";
        ctx.fillText(badge, x + 34 * scale, y + height / 2 + scale);
      }
      return canvas.toDataURL("image/png");
    },
    { png, caption, badge, scale },
  );
  return Buffer.from(url.slice(url.indexOf(",") + 1), "base64");
}

function makeGif(frames, dir, output, width) {
  const list = ["ffconcat version 1.0"];
  for (const frame of frames) list.push(`file '${frame.file}'`, `duration ${(frame.ms / 1000).toFixed(3)}`);
  // The concat demuxer drops the last duration unless the last file is listed again.
  list.push(`file '${frames[frames.length - 1].file}'`);
  const listFile = path.join(dir, "frames.txt");
  fs.writeFileSync(listFile, `${list.join("\n")}\n`);
  const filter =
    `scale=${width}:-1:flags=lanczos,split[a][b];` +
    "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle";
  const args = ["-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listFile];
  args.push("-vf", filter, "-fps_mode", "vfr", "-loop", "0", output);
  const result = spawnSync("ffmpeg", args, { stdio: "inherit" });
  if (result.error) throw new Error(`ffmpeg is needed to make the GIF: ${result.error.message}`);
  if (result.status !== 0) throw new Error(`ffmpeg exited with ${result.status}`);
}

// ---- Main --------------------------------------------------------------------------------------

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const scenarioFile = path.resolve(opts.scenario ?? path.join(here, "scenario.yaml"));
  const scenario = YAML.parse(fs.readFileSync(scenarioFile, "utf8"));
  const baseDir = path.dirname(scenarioFile);
  const outDir = path.resolve(opts.out ?? path.join(repo, "docs", "images"));
  const name = opts.name ?? scenario.name ?? "walkthrough";
  const width = opts.width ?? 960;
  const steps = scenario.steps ?? [];
  const languages = opts.lang
    ? opts.lang.split(",").map((l) => l.trim())
    : [...new Set(steps.flatMap((s) => Object.keys(s.caption ?? {})))];
  if (languages.length === 0) fail("The scenario has no captions");

  const realDir = path.resolve(opts.dataDir || process.env.TINYLM_DATA_DIR || defaultDataDir());
  const demoDir = path.join(realDir, "demo");
  const vars = { ...(scenario.vars ?? {}), ...opts.vars };
  const apiTeacher = resolveTeacher(vars, path.join(realDir, "models"));
  const models = interpolate(scenario.models ?? [], vars);
  if (vars.teacher && !apiTeacher) models.push(vars.teacher);
  console.log(`Data: ${demoDir} (models from ${realDir})`);
  if ("teacher" in (scenario.vars ?? {}))
    console.log(vars.teacher ? `Teacher: ${vars.teacher_name}` : `No teacher: examples come from ${vars.examples}`);
  prepareDataDir(demoDir, realDir, [...new Set(models)], opts.keep);
  const frameDir = path.join(demoDir, "frames");
  fs.rmSync(frameDir, { recursive: true, force: true });
  fs.mkdirSync(frameDir, { recursive: true });

  const env = { ...process.env, TINYLM_DATA_DIR: demoDir, XDG_CONFIG_HOME: path.join(demoDir, "config") };
  delete env.TINYLM_ENABLE_ECHO;
  const app = await _electron.launch({
    executablePath: electronPath,
    args: [
      desktop,
      `--user-data-dir=${path.join(demoDir, "config", "tinylm-studio-desktop")}`,
      // Chromium refuses to run as root (in a container, say) with its sandbox.
      ...(process.getuid?.() === 0 ? ["--no-sandbox"] : []),
    ],
    cwd: desktop,
    env,
  });
  const page = await app.firstWindow();
  const recorder = new Recorder(page, frameDir);
  try {
    const size = scenario.window ?? { width: 1100, height: 750 };
    await app.evaluate(({ BrowserWindow }, size) => {
      BrowserWindow.getAllWindows()[0].setContentSize(size.width, size.height);
    }, size);

    console.log("Waiting for the engine…");
    let engine;
    const deadline = Date.now() + 10 * 60_000;
    for (;;) {
      engine = await page.evaluate(() => window.tinylm.getEngineState());
      if (engine.phase === "ready") break;
      if (engine.phase === "error") throw new Error(`The engine did not start: ${engine.message}`);
      if (Date.now() > deadline) throw new Error(`The engine is not ready: ${JSON.stringify(engine)}`);
      await sleep(1000);
    }
    if (apiTeacher) {
      const response = await fetch(`${engine.url}/api/teachers`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(engine.apiKey ? { Authorization: `Bearer ${engine.apiKey}` } : {}) },
        body: JSON.stringify(apiTeacher),
      });
      if (!response.ok) throw new Error(`Could not add the teacher: ${response.status} ${await response.text()}`);
    }
    // The engine's address and API key, for scenarios that call it (Swagger UI, say).
    vars.engine_url = engine.url;
    vars.api_key = engine.apiKey ?? "";
    await sleep(1000);
    await recorder.injectCursor();

    const captions = [];
    for (const raw of steps) {
      if (raw.when && !vars[raw.when]) continue;
      if (raw.unless && vars[raw.unless]) continue;
      const step = interpolate(raw, vars);
      if (step.record === false) {
        console.log(`(preparing) ${step.caption?.en ?? ""}`);
        recorder.recording = false;
        for (const action of step.actions ?? []) await recorder.run(action, baseDir);
        recorder.recording = true;
        continue;
      }
      recorder.step = captions.length;
      captions.push(step.caption ?? {});
      console.log(`${captions.length}. ${step.caption?.en ?? Object.values(step.caption ?? {})[0] ?? ""}`);
      const first = recorder.frames.length;
      await recorder.highlight(null);
      await recorder.shot(300);
      for (const action of step.actions ?? []) await recorder.run(action, baseDir);
      const shown = recorder.frames.slice(first).reduce((sum, f) => sum + f.ms, 0);
      const needed = readingMs(step.caption ?? {});
      if (shown < needed) await recorder.hold(needed - shown);
    }

    const scale = await page.evaluate(() => window.devicePixelRatio);
    const [compositor] = await Promise.all([
      app.waitForEvent("window"),
      app.evaluate(({ BrowserWindow }) => {
        new BrowserWindow({ show: false }).loadURL("about:blank");
      }),
    ]);
    fs.mkdirSync(outDir, { recursive: true });
    for (const lang of languages) {
      const dir = path.join(frameDir, lang);
      fs.mkdirSync(dir, { recursive: true });
      const label = scenario.labels?.fast_forward?.[lang] ?? scenario.labels?.fast_forward?.en ?? "×{speed}";
      const frames = [];
      for (const [i, frame] of recorder.frames.entries()) {
        const caption = captions[frame.step][lang] ?? captions[frame.step].en ?? "";
        const badge = frame.speed ? label.replace("{speed}", frame.speed) : null;
        const file = path.join(dir, `${String(i).padStart(5, "0")}.png`);
        fs.writeFileSync(file, await compose(compositor, frame, caption, badge, scale));
        frames.push({ file, ms: frame.ms });
      }
      const output = path.join(outDir, `${name}-${lang}.gif`);
      makeGif(frames, dir, output, width);
      const seconds = frames.reduce((sum, f) => sum + f.ms, 0) / 1000;
      const mb = fs.statSync(output).size / 1e6;
      console.log(`${output}: ${frames.length} frames, ${seconds.toFixed(0)} s, ${mb.toFixed(1)} MB`);
    }
  } catch (error) {
    const file = path.join(demoDir, "error.png");
    await page.screenshot({ path: file }).catch(() => {});
    console.error(`Failed: ${error instanceof Error ? error.message : error}\nThe screen at that moment: ${file}`);
    process.exitCode = 1;
  } finally {
    await app.close().catch(() => {});
  }
}

await main();
