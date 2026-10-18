import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "./Icon";

// A spinner shows only once an operation has taken this long, so a quick one does not flicker.
const SHOW_AFTER_MS = 250;

/** A spinning icon and what is going on, e.g. "Loading documents…". */
export function Spinner({ label, block = false }: { label: string; block?: boolean }) {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setShown(true), SHOW_AFTER_MS);
    return () => clearTimeout(timer);
  }, []);
  if (!shown) return null;
  return (
    <span className={block ? "spinner block" : "spinner"} role="status">
      <Icon name="loader-2" spin />
      {label}
    </span>
  );
}

/** The long operations of a screen: loading a list, importing, exporting. */
export interface Busy {
  /** Something is running. */
  busy: boolean;
  /** What the newest running operation says it is doing. */
  label: string | null;
  /** Runs work with a label to show meanwhile; work can change it to show its progress. */
  run<T>(label: string, work: (progress: (label: string) => void) => Promise<T>): Promise<T>;
}

export function useBusy(): Busy {
  const [running, setRunning] = useState<{ id: number; label: string }[]>([]);
  const nextId = useRef(0);
  const run = useCallback(
    async <T,>(label: string, work: (progress: (label: string) => void) => Promise<T>): Promise<T> => {
      const id = ++nextId.current;
      let done = false;
      const progress = (text: string) => {
        if (!done) setRunning((list) => list.map((task) => (task.id === id ? { id, label: text } : task)));
      };
      setRunning((list) => [...list, { id, label }]);
      try {
        return await work(progress);
      } finally {
        done = true;
        setRunning((list) => list.filter((task) => task.id !== id));
      }
    },
    [],
  );
  return { busy: running.length > 0, label: running.at(-1)?.label ?? null, run };
}

/** The spinner of a screen's operations, while one runs. */
export function BusySpinner({ busy, block = false }: { busy: Busy; block?: boolean }) {
  return busy.label ? <Spinner label={busy.label} block={block} /> : null;
}
