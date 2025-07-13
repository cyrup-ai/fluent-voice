import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        worklet: "src/universal-audio.ts"
      }
    }
  }
});
