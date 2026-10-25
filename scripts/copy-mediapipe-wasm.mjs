// Copies the MediaPipe Tasks Vision WASM runtime from node_modules into
// public/mediapipe/wasm so it is self-hosted and always matches the installed
// @mediapipe/tasks-vision JS package version.
// Also applies browser compatibility patches for WebKit/macOS:
// 1. Replaces 1x1 OffscreenCanvas fallback with DOM canvas to avoid WebGL context extension crashes (gl_context.cc).
// 2. Filters Emscripten stderr INFO messages so Next.js dev overlay does not display them as console errors.
// 3. Null-guards WebGL context extension queries in the WASM JS loaders.
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const src = path.join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const dest = path.join(root, "public", "mediapipe", "wasm");

if (!existsSync(src)) {
  console.error(`[copy-mediapipe-wasm] Source not found: ${src}`);
  process.exit(1);
}

mkdirSync(dest, { recursive: true });
cpSync(src, dest, { recursive: true });
console.log(`[copy-mediapipe-wasm] Copied WASM runtime to ${dest}`);

// 1. Patch copied WASM JS loaders in public/mediapipe/wasm
const wasmJsFiles = [
  "vision_wasm_internal.js",
  "vision_wasm_nosimd_internal.js",
  "vision_wasm_module_internal.js",
];

for (const fileName of wasmJsFiles) {
  const filePath = path.join(dest, fileName);
  if (!existsSync(filePath)) continue;

  let code = readFileSync(filePath, "utf8");

  // Route Emscripten info/warning logs (e.g. TFLite XNNPACK delegate info, benign GL context info) to console.info
  // This prevents Next.js dev server from flagging them as fatal console errors.
  const oldErrPattern = "var err = console.error.bind(console);";
  const newErrCode =
    'var err = (...args) => { const msg = String(args[0] || ""); if (msg.startsWith("INFO:") || msg.includes("XNNPACK") || msg.includes("gl_context.cc") || msg.includes("GL version mismatch")) { console.info(...args); return; } console.error(...args); };';
  if (code.includes(oldErrPattern)) {
    code = code.replace(oldErrPattern, newErrCode);
  }

  // Guard getEmscriptenSupportedExtensions against null/uninitialized WebGL context
  const oldExtPattern = "var getEmscriptenSupportedExtensions = ctx => {";
  const newExtCode =
    'var getEmscriptenSupportedExtensions = ctx => { if (!ctx || typeof ctx.getSupportedExtensions !== "function") return [];';
  if (code.includes(oldExtPattern)) {
    code = code.replace(oldExtPattern, newExtCode);
  }

  // Guard glGetString(GL_EXTENSIONS = 7939) against null GLctx
  const oldGlGetExtPattern = 'case 7939:\n      ret = stringToNewUTF8(webglGetExtensions().join(" "));';
  const newGlGetExtCode =
    'case 7939:\n      ret = stringToNewUTF8((GLctx ? webglGetExtensions() : []).join(" "));';
  if (code.includes(oldGlGetExtPattern)) {
    code = code.replace(oldGlGetExtPattern, newGlGetExtCode);
  }

  writeFileSync(filePath, code, "utf8");
}
console.log("[copy-mediapipe-wasm] Patched public WASM JS loaders with GL/stderr guards.");

// 2. Patch vision_bundle.mjs & vision_bundle.cjs in node_modules to avoid new OffscreenCanvas(1,1)
const bundleFiles = [
  path.join(root, "node_modules", "@mediapipe", "tasks-vision", "vision_bundle.mjs"),
  path.join(root, "node_modules", "@mediapipe", "tasks-vision", "vision_bundle.cjs"),
];

for (const bundlePath of bundleFiles) {
  if (!existsSync(bundlePath)) continue;
  let bundleCode = readFileSync(bundlePath, "utf8");

  // Replace new OffscreenCanvas(1,1) with DOM canvas fallback on browser
  const offscreenPattern = /new OffscreenCanvas\(1,\s*1\)/g;
  if (offscreenPattern.test(bundleCode)) {
    bundleCode = bundleCode.replace(
      offscreenPattern,
      '(typeof document !== "undefined" ? document.createElement("canvas") : new OffscreenCanvas(1,1))'
    );
  }

  // In vision_bundle.mjs, force Qo() to return false so MediaPipe defaults to standard DOM canvas instead of OffscreenCanvas
  const qoMatch = /function Qo\(\)\{var t=navigator;return"undefined"!=typeof OffscreenCanvas[^}]+\}/;
  if (qoMatch.test(bundleCode)) {
    bundleCode = bundleCode.replace(qoMatch, "function Qo(){return false}");
  }

  // In vision_bundle.cjs, force ta() to return false
  const taMatch = /function ta\(\)\{var t=navigator;return"undefined"!=typeof OffscreenCanvas[^}]+\}/;
  if (taMatch.test(bundleCode)) {
    bundleCode = bundleCode.replace(taMatch, "function ta(){return false}");
  }

  writeFileSync(bundlePath, bundleCode, "utf8");
}
console.log("[copy-mediapipe-wasm] Patched Tasks Vision bundles to use standard DOM canvas.");
