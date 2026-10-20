import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The renderer is loaded from disk by Electron, so asset paths must be relative.
export default defineConfig({
  root: "src/renderer",
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../../dist/renderer",
    emptyOutDir: true,
    // The CSP allows images only from the app itself, so SVGs (the logo) stay files, not data: URLs.
    assetsInlineLimit: (file) => (file.endsWith(".svg") ? false : undefined),
  },
});
