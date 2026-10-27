import JSZip from "jszip";
import { saveAs } from "file-saver";
import {
  CapturedSample,
  PoseClassMeta,
  TrainedModelRecord,
} from "@/types/handpose";
import {
  extractFeatureVector,
  FEATURE_VECTOR_SIZE,
  formatHandAnnotationsForExport,
  getClassColor,
  LANDMARK_NAMES,
  sortDetectedHands,
} from "@/lib/landmarks";
import { base64ToUint8Array } from "@/lib/classifier";

/**
 * Generates CSV content containing both raw normalized (x,y,z) and
 * wrist-centered scale-invariant (nx,ny,nz) 21-landmark coordinates.
 */
export function buildLandmarksCsvString(samples: CapturedSample[]): string {
  const headerCols = [
    "sample_id",
    "class_name",
    "timestamp",
    "hands_count",
    "primary_handedness",
    "primary_confidence",
  ];

  for (const name of LANDMARK_NAMES) {
    headerCols.push(`h1_${name}_x`, `h1_${name}_y`, `h1_${name}_z`);
  }
  for (const name of LANDMARK_NAMES) {
    headerCols.push(`h1_${name}_norm_x`, `h1_${name}_norm_y`, `h1_${name}_norm_z`);
  }
  for (const name of LANDMARK_NAMES) {
    headerCols.push(`h2_${name}_norm_x`, `h2_${name}_norm_y`, `h2_${name}_norm_z`);
  }
  headerCols.push("inter_hand_offset_x", "inter_hand_offset_y", "inter_hand_offset_z");
  headerCols.push("hand1_present", "hand2_present");

  const rows: string[] = [headerCols.join(",")];

  const sanitizeCsvField = (val: string): string => {
    let clean = val.replace(/"/g, '""');
    if (/^[=+@-]/i.test(clean)) {
      clean = `'${clean}`;
    }
    return `"${clean}"`;
  };

  for (const sample of samples) {
    if (!sample.hands || sample.hands.length === 0) continue;
    const validHands = sample.hands.filter((h) => h.landmarks && h.landmarks.length >= 21);
    if (validHands.length === 0) continue;
    const sortedHands = sortDetectedHands(validHands);
    const primary = sortedHands[0];
    const normVec = extractFeatureVector(sortedHands) || new Array(FEATURE_VECTOR_SIZE).fill(0);

    const row: (string | number)[] = [
      sanitizeCsvField(sample.id),
      sanitizeCsvField(sample.className),
      sample.timestamp,
      validHands.length,
      primary.handedness,
      Number(primary.score.toFixed(4)),
    ];

    for (const pt of primary.landmarks) {
      row.push(
        Number(pt.x.toFixed(6)),
        Number(pt.y.toFixed(6)),
        Number(pt.z.toFixed(6))
      );
    }

    for (const val of normVec) {
      row.push(Number(val.toFixed(6)));
    }

    rows.push(row.join(","));
  }

  return rows.join("\n");
}

/**
 * Downloads the dataset as a structured ZIP archive:
 * - handpose-dataset/{className}/{className}-{id}.jpeg
 * - handpose-dataset/{className}/{className}-{id}.json
 * - handpose-dataset/landmarks.csv
 * - handpose-dataset/dataset-manifest.json
 */
export async function exportDatasetZip(
  classes: PoseClassMeta[],
  samples: CapturedSample[],
  datasetName = "handpose-dataset"
) {
  const zip = new JSZip();
  const safeRoot =
    datasetName.trim().replace(/[^a-zA-Z0-9_-]/g, "-") || "handpose-dataset";
  const root = zip.folder(safeRoot);
  if (!root) return;

  for (const [idx, sample] of samples.entries()) {
    const safeClass =
      sample.className.trim().replace(/[^a-zA-Z0-9_-]/g, "_") || "unlabeled";
    const classFolder = root.folder(safeClass);
    if (!classFolder) continue;

    const shortId = sample.id.slice(0, 8);
    const baseName = `${safeClass}-${idx+1}_${shortId}`;

    if (sample.imageDataUrl && sample.imageDataUrl.includes(",")) {
      const [header, base64Img] = sample.imageDataUrl.split(",");
      let ext = "jpeg";
      if (header.includes("image/png")) ext = "png";
      else if (header.includes("image/webp")) ext = "webp";
      classFolder.file(`${baseName}.${ext}`, base64Img, { base64: true });
    }

    const annotationPayload = formatHandAnnotationsForExport(
      sample.hands,
      sample.width,
      sample.height
    );
    classFolder.file(
      `${baseName}.json`,
      JSON.stringify(annotationPayload, null, 2)
    );
  }

  // Add tabular CSV and full JSON manifest
  root.file("landmarks.csv", buildLandmarksCsvString(samples));
  root.file(
    "dataset-manifest.json",
    JSON.stringify(
      {
        name: datasetName,
        exportedAt: new Date().toISOString(),
        totalSamples: samples.length,
        classes,
      },
      null,
      2
    )
  );

  const blob = await zip.generateAsync({ type: "blob" });
  saveAs(blob, `${safeRoot}-${Date.now().toString().slice(-6)}.zip`);
}

/**
 * Downloads only the landmarks CSV file for direct use in Python / R / ML notebooks.
 */
export function exportLandmarksCsv(
  samples: CapturedSample[],
  datasetName = "handpose-landmarks"
) {
  const safeName =
    datasetName.trim().replace(/[^a-zA-Z0-9_-]/g, "-") || "handpose-landmarks";
  const csv = buildLandmarksCsvString(samples);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  saveAs(blob, `${safeName}-${Date.now().toString().slice(-6)}.csv`);
}

/**
 * Exports a lossless JSON backup of classes and samples so it can be re-imported into the studio.
 */
export function exportDatasetJson(
  classes: PoseClassMeta[],
  samples: CapturedSample[],
  datasetName = "handpose-workspace"
) {
  const safeName =
    datasetName.trim().replace(/[^a-zA-Z0-9_-]/g, "-") || "handpose-workspace";
  const payload = {
    version: "2.0",
    name: safeName,
    exportedAt: Date.now(),
    classes,
    samples,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  saveAs(blob, `${safeName}-${Date.now().toString().slice(-6)}.json`);
}

/**
 * Parses and validates an imported JSON workspace file.
 */
export function parseImportedDatasetJson(jsonText: string): {
  classes: PoseClassMeta[];
  samples: CapturedSample[];
  name?: string;
} {
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch {
    throw new Error("Invalid JSON: Failed to parse file.");
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Invalid dataset: Root must be a JSON object.");
  }
  const raw = parsed as Record<string, unknown>;
  if (!Array.isArray(raw.samples)) {
    throw new Error("Invalid dataset JSON: missing 'samples' array.");
  }

  // Validate each sample and its 21 landmarks
  const validSamples: CapturedSample[] = [];
  for (let i = 0; i < raw.samples.length; i++) {
    const s = raw.samples[i];
    if (!s || typeof s !== "object") continue;
    const className = typeof s.className === "string" ? s.className.trim() : "";
    if (!className || !Array.isArray(s.hands) || s.hands.length === 0) continue;

    const validHands = s.hands
      .filter((h: unknown) => h && typeof h === "object")
      .map((h: Record<string, unknown>) => {
        const handedness = h.handedness === "Left" ? "Left" : "Right";
        const score = typeof h.score === "number" && !isNaN(h.score) ? h.score : 0.9;
        const landmarks = Array.isArray(h.landmarks)
          ? h.landmarks
              .filter((lm: unknown) => lm && typeof lm === "object")
              .map((lm: Record<string, unknown>) => ({
                x: typeof lm.x === "number" && !isNaN(lm.x) ? lm.x : 0,
                y: typeof lm.y === "number" && !isNaN(lm.y) ? lm.y : 0,
                z: typeof lm.z === "number" && !isNaN(lm.z) ? lm.z : 0,
              }))
          : [];
        const worldLandmarks = Array.isArray(h.worldLandmarks)
          ? h.worldLandmarks
              .filter((lm: unknown) => lm && typeof lm === "object")
              .map((lm: Record<string, unknown>) => ({
                x: typeof lm.x === "number" && !isNaN(lm.x) ? lm.x : 0,
                y: typeof lm.y === "number" && !isNaN(lm.y) ? lm.y : 0,
                z: typeof lm.z === "number" && !isNaN(lm.z) ? lm.z : 0,
              }))
          : landmarks.map((p) => ({
              x: (p.x - 0.5) * 0.2,
              y: (p.y - 0.5) * 0.2,
              z: p.z * 0.2,
            }));
        return { handedness, score, landmarks, worldLandmarks };
      })
      .filter((h: { landmarks: unknown[] }) => h.landmarks.length === 21);

    if (validHands.length === 0) continue;

    const rawDataUrl = typeof s.imageDataUrl === "string" ? s.imageDataUrl.trim() : "";
    const isSafeDataUrl = /^data:image\/(jpe?g|png|webp);base64,[A-Za-z0-9+/=\s\-_]+$/i.test(rawDataUrl);
    const imageDataUrl = isSafeDataUrl ? rawDataUrl : "";

    validSamples.push({
      id: typeof s.id === "string" && s.id ? s.id : `sample_${Date.now()}_${i}`,
      className,
      timestamp: typeof s.timestamp === "number" ? s.timestamp : Date.now(),
      imageDataUrl,
      hands: validHands,
      width: typeof s.width === "number" ? s.width : 640,
      height: typeof s.height === "number" ? s.height : 480,
    });
  }

  if (validSamples.length === 0 && raw.samples.length > 0) {
    throw new Error("Invalid dataset: No samples contained valid 21-point hand landmark data.");
  }

  // Validate classes and auto-reconstruct any missing classes found in samples
  const rawClasses = Array.isArray(raw.classes) ? raw.classes : [];
  const knownClassNames = new Set<string>();
  const validClasses: PoseClassMeta[] = [];

  for (const c of rawClasses) {
    if (c && typeof c === "object" && typeof c.name === "string" && c.name.trim()) {
      const name = c.name.trim();
      if (!knownClassNames.has(name.toLowerCase())) {
        knownClassNames.add(name.toLowerCase());
        validClasses.push({
          name,
          color: typeof c.color === "string" && c.color ? c.color : getClassColor(validClasses.length),
          createdAt: typeof c.createdAt === "number" ? c.createdAt : Date.now(),
        });
      }
    }
  }

  for (const s of validSamples) {
    if (!knownClassNames.has(s.className.toLowerCase())) {
      knownClassNames.add(s.className.toLowerCase());
      validClasses.push({
        name: s.className,
        color: getClassColor(validClasses.length),
        createdAt: Date.now(),
      });
    }
  }

  // Normalize each sample's className to canonical casing registered in validClasses
  const canonicalNameMap = new Map<string, string>();
  for (const c of validClasses) {
    canonicalNameMap.set(c.name.toLowerCase(), c.name);
  }

  const normalizedSamples = validSamples.map((s) => {
    const canonical = canonicalNameMap.get(s.className.toLowerCase());
    return canonical && canonical !== s.className
      ? { ...s, className: canonical }
      : s;
  });

  return {
    classes: validClasses,
    samples: normalizedSamples,
    name: typeof raw.name === "string" ? raw.name : undefined,
  };
}

/**
 * Exports a trained TensorFlow.js model as a ready-to-use ZIP archive containing:
 * - model.json (topology + weightsManifest)
 * - model.weights.bin (raw binary float32 weights)
 * - classes.json (ordered class labels + metadata)
 * - inference-example.js (sample code showing how to load and run the model)
 */
export async function exportTrainedModelZip(record: TrainedModelRecord) {
  if (!record.serializedModel) {
    throw new Error("No serialized TensorFlow.js weights available for export.");
  }

  const zip = new JSZip();
  const { modelTopology, weightSpecs, weightDataBase64 } =
    record.serializedModel;

  const modelJson = {
    format: "layers-model",
    generatedBy: "Hand Pose Studio v2.0",
    convertedBy: null,
    modelTopology,
    weightsManifest: [
      {
        paths: ["model.weights.bin"],
        weights: weightSpecs,
      },
    ],
  };

  const weightBytes = base64ToUint8Array(weightDataBase64);

  zip.file("model.json", JSON.stringify(modelJson, null, 2));
  zip.file("model.weights.bin", weightBytes);
  zip.file(
    "classes.json",
    JSON.stringify(
      {
        id: record.id,
        name: record.name,
        classes: record.classes,
        finalAccuracy: record.finalAccuracy,
        trainedAt: new Date(record.trainedAt).toISOString(),
        inputShape: [1, FEATURE_VECTOR_SIZE],
        preprocessing:
          "Normalized 131-D feature vector: Hand 1 (63), Hand 2 (63), relative wrist offset (3), presence flags (2). Supports both 1-hand and 2-hand gestures.",
      },
      null,
      2
    )
  );

  const exampleCode = `// Example: Loading and running your Hand Pose Studio TF.js model
import * as tf from '@tensorflow/tfjs';

const CLASSES = ${JSON.stringify(record.classes)};
const FEATURE_DIM = ${FEATURE_VECTOR_SIZE};

export async function loadPoseClassifier() {
  const model = await tf.loadLayersModel('./model.json');
  return model;
}

function normalizeSingleHand(landmarks21) {
  const wrist = landmarks21[0];
  const centered = landmarks21.map(p => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));
  const maxDist = Math.max(...centered.map(p => Math.hypot(p.x, p.y, p.z)), 1e-6);
  return {
    features: centered.flatMap(p => [p.x / maxDist, p.y / maxDist, p.z / maxDist]),
    scale: maxDist,
  };
}

function sortDetectedHands(hands) {
  if (hands.length <= 1) return hands;
  const left = hands.find(h => h.handedness === 'Left');
  const right = hands.find(h => h.handedness === 'Right');
  if (left && right) return [left, right];
  return [...hands].sort((a, b) => (a.landmarks[0]?.x ?? 0) - (b.landmarks[0]?.x ?? 0));
}

export function extractFeatureVector(detectedHands) {
  if (!detectedHands || detectedHands.length === 0) return null;
  const validHands = detectedHands.filter(h => h.landmarks && h.landmarks.length >= 21);
  if (validHands.length === 0) return null;

  const sorted = sortDetectedHands(validHands);
  const out = new Array(FEATURE_DIM).fill(0);
  const hand1 = sorted[0];
  const norm1 = normalizeSingleHand(hand1.landmarks);
  for (let i = 0; i < 63; i++) out[i] = norm1.features[i];
  out[129] = 1.0;

  if (sorted.length > 1) {
    const hand2 = sorted[1];
    const norm2 = normalizeSingleHand(hand2.landmarks);
    for (let i = 0; i < 63; i++) out[63 + i] = norm2.features[i];
    const w1 = hand1.landmarks[0];
    const w2 = hand2.landmarks[0];
    out[126] = (w2.x - w1.x) / norm1.scale;
    out[127] = (w2.y - w1.y) / norm1.scale;
    out[128] = (w2.z - w1.z) / norm1.scale;
    out[130] = 1.0;
  }
  return out;
}

export function predictPose(model, detectedHands) {
  const features = extractFeatureVector(detectedHands);
  if (!features) return null;
  return tf.tidy(() => {
    const input = tf.tensor2d([features], [1, FEATURE_DIM]);
    const probs = model.predict(input).dataSync();
    return CLASSES.map((name, i) => ({ className: name, probability: probs[i] }))
      .sort((a, b) => b.probability - a.probability);
  });
}
`;

  zip.file("inference-example.js", exampleCode);

  const blob = await zip.generateAsync({ type: "blob" });
  const safeName =
    record.name.trim().replace(/[^a-zA-Z0-9_-]/g, "-") || "handpose-tfjs-model";
  saveAs(blob, `${safeName}.zip`);
}
