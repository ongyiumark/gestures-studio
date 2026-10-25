import JSZip from "jszip";
import { saveAs } from "file-saver";
import {
  CapturedSample,
  PoseClassMeta,
  TrainedModelRecord,
} from "@/types/handpose";
import {
  extractFeatureVector,
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
    "handedness",
    "confidence_score",
  ];

  for (const name of LANDMARK_NAMES) {
    headerCols.push(`${name}_x`, `${name}_y`, `${name}_z`);
  }
  for (const name of LANDMARK_NAMES) {
    headerCols.push(`${name}_norm_x`, `${name}_norm_y`, `${name}_norm_z`);
  }

  const rows: string[] = [headerCols.join(",")];

  for (const sample of samples) {
    if (!sample.hands || sample.hands.length === 0) continue;
    const primary = sample.hands[0];
    const normVec = extractFeatureVector(sample.hands) || new Array(63).fill(0);

    const row: (string | number)[] = [
      `"${sample.id}"`,
      `"${sample.className.replace(/"/g, '""')}"`,
      sample.timestamp,
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
    generatedBy: "Gesture Studio v2.0",
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
        inputShape: [1, 63],
        preprocessing:
          "Wrist-centered (landmark 0 subtracted) and divided by max 3D Euclidean distance from wrist.",
      },
      null,
      2
    )
  );

  const exampleCode = `// Example: Loading and running your Hand Pose Studio TF.js model
import * as tf from '@tensorflow/tfjs';

const CLASSES = ${JSON.stringify(record.classes)};

export async function loadPoseClassifier() {
  const model = await tf.loadLayersModel('./model.json');
  return model;
}

export function normalizeLandmarks(landmarks21) {
  const wrist = landmarks21[0];
  const centered = landmarks21.map(p => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));
  const maxDist = Math.max(...centered.map(p => Math.hypot(p.x, p.y, p.z)), 1e-6);
  return centered.flatMap(p => [p.x / maxDist, p.y / maxDist, p.z / maxDist]);
}

export function predictPose(model, landmarks21) {
  const features = normalizeLandmarks(landmarks21);
  return tf.tidy(() => {
    const input = tf.tensor2d([features], [1, 63]);
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
