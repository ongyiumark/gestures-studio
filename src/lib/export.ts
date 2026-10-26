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
  LANDMARK_NAMES,
} from "@/lib/landmarks";

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

  for (const sample of samples) {
    if (!sample.hands || sample.hands.length === 0) continue;
    const primary = sample.hands[0];
    const normVec = extractFeatureVector(sample.hands) || new Array(FEATURE_VECTOR_SIZE).fill(0);

    const row: (string | number)[] = [
      `"${sample.id}"`,
      `"${sample.className.replace(/"/g, '""')}"`,
      sample.timestamp,
      sample.hands.length,
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

  for (const sample of samples) {
    const safeClass =
      sample.className.trim().replace(/[^a-zA-Z0-9_-]/g, "_") || "unlabeled";
    const classFolder = root.folder(safeClass);
    if (!classFolder) continue;

    const shortId = sample.id.slice(0, 8);
    const baseName = `${safeClass}-${shortId}`;

    if (sample.imageDataUrl && sample.imageDataUrl.includes(",")) {
      const base64Img = sample.imageDataUrl.split(",")[1];
      classFolder.file(`${baseName}.jpeg`, base64Img, { base64: true });
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
  const csv = buildLandmarksCsvString(samples);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  saveAs(blob, `${datasetName}-${Date.now().toString().slice(-6)}.csv`);
}

/**
 * Exports a lossless JSON backup of classes and samples so it can be re-imported into the studio.
 */
export function exportDatasetJson(
  classes: PoseClassMeta[],
  samples: CapturedSample[],
  datasetName = "handpose-workspace"
) {
  const payload = {
    version: "2.0",
    name: datasetName,
    exportedAt: Date.now(),
    classes,
    samples,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  saveAs(blob, `${datasetName}-${Date.now().toString().slice(-6)}.json`);
}

/**
 * Parses and validates an imported JSON workspace file.
 */
export function parseImportedDatasetJson(jsonText: string): {
  classes: PoseClassMeta[];
  samples: CapturedSample[];
  name?: string;
} {
  const parsed = JSON.parse(jsonText);
  if (!parsed || !Array.isArray(parsed.samples)) {
    throw new Error("Invalid dataset JSON: missing 'samples' array.");
  }
  const classes: PoseClassMeta[] = Array.isArray(parsed.classes)
    ? parsed.classes
    : [];
  const samples: CapturedSample[] = parsed.samples;
  return {
    classes,
    samples,
    name: typeof parsed.name === "string" ? parsed.name : undefined,
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

  const binary = atob(weightDataBase64);
  const weightBytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    weightBytes[i] = binary.charCodeAt(i);
  }

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

export function extractFeatureVector(detectedHands) {
  if (!detectedHands || detectedHands.length === 0) return null;
  const out = new Array(FEATURE_DIM).fill(0);
  const hand1 = detectedHands[0];
  const norm1 = normalizeSingleHand(hand1.landmarks);
  for (let i = 0; i < 63; i++) out[i] = norm1.features[i];
  out[129] = 1.0;

  if (detectedHands.length > 1) {
    const hand2 = detectedHands[1];
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
