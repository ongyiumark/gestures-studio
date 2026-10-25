# Gesture Studio 2.0

An interactive 3D hand-pose dataset builder, in-browser neural network trainer, and real-time gesture classification studio built with **Next.js (App Router)**, **TypeScript**, **Tailwind CSS**, **Google MediaPipe HandLandmarker**, and **TensorFlow.js**.

## Key Features

- **Modern 3D Hand Tracking (`@mediapipe/tasks-vision`)**:
  - Real-time 21-keypoint 2D normalized & 3D metric world landmark detection via WebAssembly SIMD and WebGL.
  - Multi-hand tracking (1 or 2 hands) with handedness (`Left` / `Right`) and confidence telemetry.
  - Self-hosted WASM runtime and offline task model.
- **Guided 2-Step Workflow + Interactive Tutorial**:
  - Built-in 4-step interactive walkthrough modal for first-time users.
  - Focused tabbed workspace (**Step 1: Collect Poses** &rarr; **Step 2: Train & Test Live**) to keep the UI clean and approachable.
- **Flexible Data Collection Modes**:
  - **Single Capture** (`Space` shortcut or button)
  - **3-Second Countdown Timer**
  - **Auto-Burst Recording** (captures 10–25 frames automatically while holding a pose)
  - **One-Click Demo Dataset** (32 synthetic 3D hand pose samples across 4 classes for instant testing without a webcam)
- **In-Browser Machine Learning (`@tensorflow/tfjs`)**:
  - **TF.js Multi-Layer Perceptron (MLP)**: Trains a 3-layer neural network on translation- and scale-invariant 63-D wrist-centered landmark vectors with 5x 3D rotation/jitter augmentation and live SVG Loss & Accuracy curves.
  - **Instant k-NN Classifier**: Zero-wait distance-weighted Nearest Neighbors classifier with Leave-One-Out accuracy evaluation.
  - **Live Webcam Prediction HUD**: Real-time gesture prediction banner with animated per-class confidence bars.
- **100% Browser Local Storage & Rich Exports**:
  - **Zero Server Storage / Complete Privacy**: All classes, captured photos, 3D landmarks, and trained models are stored strictly in the user's browser (`localStorage` + browser `IndexedDB`). Nothing is ever uploaded to a server.
  - **Export Formats**:
    - **Full Dataset Archive (`.zip`)**: Class folders with `.jpeg` frames, `.json` annotations (both MediaPipe and legacy `@tensorflow-models/handpose` format), `landmarks.csv`, and `dataset-manifest.json`.
    - **Landmarks Feature Table (`.csv`)**: Flat tabular CSV with raw & normalized 63-D coordinates ready for Python, `scikit-learn`, or PyTorch.
    - **Trained TF.js Model Bundle (`.zip`)**: Exports `model.json`, `model.weights.bin`, `classes.json`, and a ready-to-run `inference-example.js`.
    - **Workspace Backup (`.json`)**: Lossless JSON export & import.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.
