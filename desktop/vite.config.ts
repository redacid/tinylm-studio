import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The renderer is loaded from disk by Electron, so asset paths must be relative.
export default defineConfig({
  root: "src/renderer",
  base: "./",
  plugins: [react()],
  build: { outDir: "../../dist/renderer", emptyOutDir: true },
});
