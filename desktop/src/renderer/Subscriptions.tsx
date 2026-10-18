import { FormEvent, useCallback, useEffect, useState } from "react";
import { CliTool, EngineClient } from "./api";
import { BusySpinner, Spinner, useBusy } from "./Busy";
import { Icon } from "./Icon";

/** What to do on each tool's login page. */
const STEPS: Record<CliTool["tool"], string> = {
  claude: "Open the link, sign in with your Claude account (Pro or Max) and paste the code the page shows below.",
  codex:
    "Open the link, sign in with your ChatGPT account and enter the one-time code below. The login finishes by itself.",
  gemini: "Open the link, sign in with your Google account and paste the authorization code the page shows below.",
};

/** Where ChatGPT turns on logging Codex in with a device code, which a login needs first. */
const CHATGPT_SECURITY = "https://chatgpt.com/#settings/Security";

/** Codex CLI's one-time code, such as ABCD-1234, from its output. */
function oneTimeCode(output: string): string | null {
  const found = output.match(/\b[A-Z0-9]{4,}-[A-Z0-9]{4,}\b/g);
  return found ? found[found.length - 1] : null;
}

/** Settings › Subscriptions: logging the engine's command-line tools in, for cli teachers. */
export function SubscriptionsForm({ client }: { client: EngineClient | null }) {
  const [tools, setTools] = useState<CliTool[]>([]);
  const [error, setError] = useState<string | null>(null);
  const loading = useBusy();
  const { run: load } = loading;

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  const refresh = useCallback(async () => {
    if (!client) return;
    try {
      setTools(await client.cliTools());
      setError(null);
    } catch (e) {
      report(e);
    }
  }, [client]);

  useEffect(() => {
    load("Checking the tools…", refresh);
  }, [load, refresh]);

  // While a login goes on, its link, output and end show up as they come; asking for the logins
  // only is quick, and the tools' status is checked again once one ends.
  const running = tools.some((t) => t.login?.state === "running");
  useEffect(() => {
    if (!running || !client) return;
    const timer = setInterval(async () => {
      try {
        const logins = await client.cliLogins();
        const ended = tools.some((t) => t.login?.state === "running" && logins[t.tool]?.state !== "running");
        setTools((list) => list.map((t) => ({ ...t, login: logins[t.tool] ?? null })));
        if (ended) await load("Checking the login…", refresh);
      } catch (e) {
        report(e);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [running, client, tools, load, refresh]);

  async function act(action: () => Promise<unknown>, label?: string) {
    setError(null);
    try {
      await action();
    } catch (e) {
      report(e);
    }
    await (label ? load(label, refresh) : refresh());
  }

  /** Starts a login: the card spins until the tool has printed its link. */
  async function login(tool: CliTool) {
    setError(null);
    try {
      const started = await client!.cliLogin(tool.tool);
      setTools((list) => list.map((t) => (t.tool === tool.tool ? { ...t, login: started } : t)));
    } catch (e) {
      report(e);
    }
  }

  if (!client) return <p className="muted">The engine is starting.</p>;
  return (
    <div className="settings-form subscriptions">
      <h3>Subscriptions</h3>
      <p className="muted">
        Teachers can answer with a Claude, ChatGPT or Google subscription instead of an API key, through its
        command-line tool on the engine's computer (Models › Teachers). Log each tool in here once: you open its link in
        your browser, and the engine keeps the login in its data folder (on the volume of a container or pod). Another
        engine needs its own login.
      </p>
      {error && <p className="error">{error}</p>}
      <BusySpinner busy={loading} block />
      {tools.map((t) => (
        <ToolCard
          key={t.tool}
          tool={t}
          onLogin={() => login(t)}
          onCode={async (code) => {
            await client.cliLoginCode(t.tool, code).catch(report);
            setTools(await client.cliTools().catch(() => tools));
          }}
          onCancel={() => act(() => client.cliLoginCancel(t.tool))}
          onLogout={() => act(() => client.cliLogout(t.tool), `Logging ${t.name} out…`)}
        />
      ))}
    </div>
  );
}

function ToolCard({
  tool: t,
  onLogin,
  onCode,
  onCancel,
  onLogout,
}: {
  tool: CliTool;
  onLogin: () => Promise<void>;
  onCode: (code: string) => Promise<void>;
  onCancel: () => void;
  onLogout: () => void;
}) {
  const [code, setCode] = useState("");
  const [confirmLogout, setConfirmLogout] = useState(false);
  // Between pressing Log in and the engine starting the tool, and sending a code.
  const [starting, setStarting] = useState(false);
  const [sending, setSending] = useState(false);
  const login = t.login;
  const running = login?.state === "running";

  async function start() {
    setStarting(true);
    try {
      await onLogin();
    } finally {
      setStarting(false);
    }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    setSending(true);
    try {
      await onCode(code);
      setCode("");
    } finally {
      setSending(false);
    }
  }

  const deviceCode = t.tool === "codex" && login ? oneTimeCode(login.output) : null;
  return (
    <div className="cli-tool">
      <div className="cli-tool-head">
        <Icon name="terminal-2" />
        <strong>{t.name}</strong>
        <span className="muted">{t.installed ? t.version ?? "installed" : "not installed on the engine's computer"}</span>
        {t.installed && (
          <span className={`badge ${t.logged_in ? "done" : ""}`}>
            {t.logged_in ? (t.account ? `logged in · ${t.account}` : "logged in") : "not logged in"}
          </span>
        )}
        <span className="cli-tool-actions">
          {starting ? (
            <Spinner label={`Starting ${t.name}…`} />
          ) : running ? (
            <button onClick={onCancel}><Icon name="x" />Cancel</button>
          ) : confirmLogout ? (
            <>
              <button className="danger" onClick={() => { setConfirmLogout(false); onLogout(); }}><Icon name="logout" />Log out</button>
              <button onClick={() => setConfirmLogout(false)}><Icon name="x" />Cancel</button>
            </>
          ) : (
            <>
              <button onClick={start} disabled={!t.installed}>
                <Icon name="login" />
                {t.logged_in ? "Log in again" : "Log in"}
              </button>
              {t.logged_in && <button onClick={() => setConfirmLogout(true)}><Icon name="logout" />Log out</button>}
            </>
          )}
        </span>
      </div>
      {t.tool === "codex" && t.installed && !t.logged_in && (
        <p className="muted cli-note">
          <Icon name="info-circle" /> First turn on device code login for Codex in ChatGPT's{" "}
          <a href={CHATGPT_SECURITY} target="_blank" rel="noreferrer">security settings</a> (Settings › Security);
          without it the login page refuses the code.
        </p>
      )}
      {running && (
        <div className="cli-login">
          <p>{STEPS[t.tool]}</p>
          {login.url ? (
            <div className="buttons-row">
              <a className="button" href={login.url} target="_blank" rel="noreferrer">
                <Icon name="external-link" />Open the login page
              </a>
              <button onClick={() => navigator.clipboard.writeText(login.url!)}><Icon name="copy" />Copy link</button>
            </div>
          ) : (
            <Spinner label={`Waiting for ${t.name} to give the login link…`} />
          )}
          {deviceCode && (
            <p>
              One-time code: <code className="device-code">{deviceCode}</code>
            </p>
          )}
          {(sending || login.checking_code) && <Spinner label="Checking the code…" />}
          {login.wants_code && !sending && (
            <form className="key-row" onSubmit={send}>
              <input
                placeholder="Code from the login page"
                autoComplete="off"
                value={code}
                onChange={(e) => setCode(e.target.value)}
              />
              <button type="submit" disabled={!code.trim()}><Icon name="send" />Send</button>
            </form>
          )}
        </div>
      )}
      {login && login.state === "failed" && <p className="error">Login failed: {login.error}</p>}
      {login && login.state === "done" && <p className="muted">Logged in.</p>}
      {login && login.output.trim() && (
        <details className="cli-output">
          <summary>The tool's output</summary>
          <pre>{login.output.trim()}</pre>
        </details>
      )}
    </div>
  );
}
