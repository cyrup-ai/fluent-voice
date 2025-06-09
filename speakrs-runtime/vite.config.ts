// vite.config.ts
import { resolve } from "path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        "ua-worklet": resolve(__dirname, "src/universal-audio.ts")
      },
      output: {
        entryFileNames: assetInfo =>
          assetInfo.name === "ua-worklet"
            ? "universal-audio.js"       // fixed filename
            : "assets/[name].[hash].js"
      }
    }
  }
});
