// Placeholder UI: shows engine status. Replaced by the React app in Phase 1.
type EngineState = { phase: string; port?: number; version?: string; message?: string };

declare global {
  interface Window {
    tinylm: {
      getEngineState(): Promise<EngineState>;
      onEngineState(callback: (state: EngineState) => void): void;
    };
  }
}

const labels: Record<string, string> = {
  "setting-up": "Setting up the Python environment…",
  starting: "Starting engine…",
  ready: "Engine ready",
  error: "Engine error",
  stopped: "Engine stopped",
};

function render(state: EngineState) {
  const status = document.getElementById("status")!;
  const details = [state.version && `v${state.version}`, state.port && `port ${state.port}`, state.message]
    .filter(Boolean)
    .join(" · ");
  status.textContent = `${labels[state.phase] ?? state.phase}${details ? ` (${details})` : ""}`;
  status.dataset.phase = state.phase;
}

window.tinylm.onEngineState(render);
window.tinylm.getEngineState().then(render);

export {};
