/** @type {import("eslint").Linter.Config} */
module.exports = {
    root: true,
    env: { browser: true, es2022: true, node: true },
    parser: "@typescript-eslint/parser",
    plugins: ["@typescript-eslint", "import"],
    extends: [
      "eslint:recommended",
      "plugin:@typescript-eslint/strict-type-checked",
      "plugin:import/recommended",
      "plugin:import/typescript"
    ],
    rules: {
      "import/no-unresolved": "off",
      "no-console": ["warn", { allow: ["warn", "error"] }]
    },
    ignorePatterns: ["dist", "node_modules"]
  };
  
  