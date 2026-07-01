import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Builds directly into ../dist (this repo's WEB_DIRECTORY target) -- no
// custom URL prefix needed, unlike the official React Extension Template
// (which registers a manual static route for a custom prefix); the plain
// WEB_DIRECTORY attribute already gives ComfyUI everything it needs to
// auto-discover and serve this, confirmed independent of this project's
// v3 comfy_entrypoint() node-listing mechanism.
export default defineConfig({
  plugins: [react()],
  build: {
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, "src/main.tsx"),
      },
      output: {
        dir: "../dist",
        entryFileNames: "[name].js",
        chunkFileNames: "[name]-[hash].js",
        assetFileNames: "[name][extname]",
        // React/ReactDOM ship bundled with the extension (ComfyUI doesn't
        // expose them as globals for extensions to consume) -- split into
        // a separate vendor chunk purely for browser caching, not because
        // anything external depends on the split.
        manualChunks: {
          vendor: ["react", "react-dom"],
        },
      },
    },
  },
});
