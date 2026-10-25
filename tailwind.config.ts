import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        ocean: {
          50: "#f0f7fc",
          100: "#dfedf8",
          200: "#c3def2",
          300: "#97c7e9",
          400: "#65a9dc",
          500: "#3d8dcc",
          600: "#286398", // Original heritage maritime blue
          700: "#215380",
          800: "#1d466b",
          900: "#1b3c5a",
          950: "#102438",
        },
        terracotta: {
          50: "#fef6ee",
          100: "#fde9d7",
          200: "#fbd1af",
          300: "#f7b27d",
          400: "#ff8a55", // Original heritage title text shadow
          500: "#ea580c",
          600: "#c74a11", // Original heritage button & scrollbar color
          700: "#9e3201",
          800: "#7c2807",
          900: "#64220b",
          950: "#390f03",
        },
        gold: {
          100: "#fff8d6",
          200: "#ffe79c", // Original heritage author link color
          300: "#ebbe30", // Original heritage link hover color
          400: "#d4a317",
          500: "#b5840d",
          600: "#916307",
        },
        surface: {
          DEFAULT: "#121e2d",
          card: "#132238",
          well: "#0a121c",
          elevated: "#182b44",
        },
      },
    },
  },
  plugins: [],
};

export default config;
