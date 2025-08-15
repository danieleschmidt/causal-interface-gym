module.exports = [
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    languageOptions: {
      parser: require("@typescript-eslint/parser"),
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: {
          jsx: true
        }
      }
    },
    plugins: {
      "@typescript-eslint": require("@typescript-eslint/eslint-plugin"),
      "react": require("eslint-plugin-react"),
      "react-hooks": require("eslint-plugin-react-hooks")
    },
    rules: {
      "@typescript-eslint/no-unused-vars": "warn",
      "react/react-in-jsx-scope": "off",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn"
    },
    settings: {
      react: {
        version: "detect"
      }
    }
  }
];