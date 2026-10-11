export type EngineState =
  | { phase: "setting-up" }
  | { phase: "starting"; port: number }
  | { phase: "ready"; port: number; version: string }
  | { phase: "error"; message: string }
  | { phase: "stopped" };

declare global {
  interface Window {
    tinylm: {
      getEngineState(): Promise<EngineState>;
      onEngineState(callback: (state: EngineState) => void): void;
    };
  }
}
