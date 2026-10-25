import * as tf from "@tensorflow/tfjs";
import {
  CapturedSample,
  DetectedHand,
  PoseClassMeta,
  PredictionResult,
  SerializedTFModel,
  TrainingConfig,
  TrainingHistoryPoint,
} from "@/types/handpose";
import {
  augmentFeatureVector,
  extractFeatureVector,
   getClassColor,
} from "@/lib/landmarks";

export interface PreparedDataset {
  features: number[][];
  labels: number[];
  classes: string[];
  rawSamplesPerClass: Record<string, number>;
}

/**
 * Extracts normalized 63-D feature vectors from captured samples,
 * optionally applying synthetic jitter/rotation augmentation.
 */
export function prepareTrainingDataset(
  samples: CapturedSample[],
  useAugmentation: boolean,
  augmentCopies = 4
): PreparedDataset {
  const validSamples = samples.filter(
    (s) => s.hands && s.hands.length > 0 && extractFeatureVector(s.hands) !== null
  );

  const classes = Array.from(new Set(validSamples.map((s) => s.className))).sort();
  const classToIndex = new Map<string, number>();
  classes.forEach((c, idx) => classToIndex.set(c, idx));

  const features: number[][] = [];
  const labels: number[] = [];
  const rawSamplesPerClass: Record<string, number> = {};

  for (const sample of validSamples) {
    const vec = extractFeatureVector(sample.hands);
    const labelIdx = classToIndex.get(sample.className);
    if (!vec || labelIdx === undefined) continue;

    rawSamplesPerClass[sample.className] =
      (rawSamplesPerClass[sample.className] || 0) + 1;

    features.push(vec);
    labels.push(labelIdx);

    if (useAugmentation) {
      for (let i = 0; i < augmentCopies; i++) {
        features.push(augmentFeatureVector(vec));
        labels.push(labelIdx);
      }
    }
  }

  return { features, labels, classes, rawSamplesPerClass };
}

/**
 * Shuffles feature & label arrays in unison using Fisher-Yates.
 */
function shuffleDataset(features: number[][], labels: number[]) {
  const indices = features.map((_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return {
    shuffledFeatures: indices.map((i) => features[i]),
    shuffledLabels: indices.map((i) => labels[i]),
  };
}

/**
 * Trains a TensorFlow.js Multi-Layer Perceptron (MLP) on 63-D normalized hand landmark vectors.
 */
export async function trainMLPClassifier(
  samples: CapturedSample[],
  config: TrainingConfig,
  onProgress: (point: TrainingHistoryPoint) => void
): Promise<{
  model: tf.LayersModel;
  classes: string[];
  history: TrainingHistoryPoint[];
  serializedModel: SerializedTFModel;
}> {
  await tf.ready();

  const prepared = prepareTrainingDataset(samples, config.useAugmentation, 4);
  const { classes } = prepared;

  if (classes.length < 2) {
    throw new Error(
      "At least 2 distinct pose classes with valid hand landmarks are required to train a classifier."
    );
  }

  const { shuffledFeatures, shuffledLabels } = shuffleDataset(
    prepared.features,
    prepared.labels
  );

  const numClasses = classes.length;
  const xs = tf.tensor2d(shuffledFeatures, [shuffledFeatures.length, 63]);
  const ys = tf.oneHot(tf.tensor1d(shuffledLabels, "int32"), numClasses);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [63],
      units: config.hiddenUnits,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(
    tf.layers.dense({
      units: Math.max(16, Math.floor(config.hiddenUnits / 2)),
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(config.learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const history: TrainingHistoryPoint[] = [];
  const validationSplit = shuffledFeatures.length >= 20 ? 0.15 : 0;

  await model.fit(xs, ys, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationSplit,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const point: TrainingHistoryPoint = {
          epoch: epoch + 1,
          loss: Number((logs?.loss ?? 0).toFixed(4)),
          accuracy: Number(((logs?.acc ?? logs?.accuracy ?? 0) * 100).toFixed(1)),
          valLoss:
            logs?.val_loss !== undefined
              ? Number(logs.val_loss.toFixed(4))
              : undefined,
          valAccuracy:
            logs?.val_acc !== undefined || logs?.val_accuracy !== undefined
              ? Number(
                  ((logs?.val_acc ?? logs?.val_accuracy ?? 0) * 100).toFixed(1)
                )
              : undefined,
        };
        history.push(point);
        onProgress(point);
        await tf.nextFrame();
      },
    },
  });

  xs.dispose();
  ys.dispose();

  const serializedModel = await serializeTFModel(model);

  return {
    model,
    classes,
    history,
    serializedModel,
  };
}

/**
 * Serializes a tf.LayersModel into a JSON-friendly object (base64 weights)
 * for IndexedDB, Backend API storage, and ZIP export.
 */
export async function serializeTFModel(
  model: tf.LayersModel
): Promise<SerializedTFModel> {
  let captured: SerializedTFModel | null = null;

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      const weightBuffer = artifacts.weightData as ArrayBuffer;
      const bytes = new Uint8Array(weightBuffer);
      let binary = "";
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const weightDataBase64 = btoa(binary);

      captured = {
        modelTopology: artifacts.modelTopology,
        weightSpecs: artifacts.weightSpecs,
        weightDataBase64,
      };

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
          weightDataBytes: bytes.byteLength,
        },
      };
    })
  );

  if (!captured) {
    throw new Error("Failed to serialize TensorFlow.js model");
  }
  return captured;
}

/**
 * Reconstructs a tf.LayersModel in memory from a SerializedTFModel object.
 */
export async function deserializeTFModel(
  serialized: SerializedTFModel
): Promise<tf.LayersModel> {
  await tf.ready();
  const binary = atob(serialized.weightDataBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: serialized.modelTopology as tf.io.ModelJSON["modelTopology"],
      weightSpecs: serialized.weightSpecs as tf.io.WeightsManifestEntry[],
      weightData: bytes.buffer,
    })
  );
}

/**
 * Predicts hand pose class probabilities using a trained TF.js LayersModel.
 */
export function predictWithMLP(
  model: tf.LayersModel,
  classes: string[],
  hands: DetectedHand[],
  classMeta: PoseClassMeta[]
): PredictionResult | null {
  const vec = extractFeatureVector(hands);
  if (!vec || classes.length === 0) return null;

  const probs = tf.tidy(() => {
    const input = tf.tensor2d([vec], [1, 63]);
    const output = model.predict(input) as tf.Tensor;
    return Array.from(output.dataSync());
  });

  const colorMap = new Map(classMeta.map((c) => [c.name, c.color]));

  const probabilities = classes
    .map((className, idx) => ({
      className,
      probability: probs[idx] ?? 0,
      color: colorMap.get(className) || getClassColor(idx),
    }))
    .sort((a, b) => b.probability - a.probability);

  return {
    className: probabilities[0].className,
    confidence: probabilities[0].probability,
    probabilities,
  };
}

/**
 * Instant k-Nearest Neighbors (k-NN) classifier over normalized 63-D landmark vectors.
 * Uses distance-weighted voting over the top k=5 neighbors.
 */
export function predictWithKNN(
  samples: CapturedSample[],
  hands: DetectedHand[],
  classMeta: PoseClassMeta[],
  k = 5
): PredictionResult | null {
  const queryVec = extractFeatureVector(hands);
  if (!queryVec) return null;

  const validSamples: { className: string; vec: number[] }[] = [];
  for (const s of samples) {
    const v = extractFeatureVector(s.hands);
    if (v) validSamples.push({ className: s.className, vec: v });
  }

  const classes = Array.from(new Set(validSamples.map((s) => s.className))).sort();
  if (classes.length < 2) return null;

  const distances = validSamples.map((item) => {
    let sumSq = 0;
    for (let i = 0; i < queryVec.length; i++) {
      const diff = queryVec[i] - item.vec[i];
      sumSq += diff * diff;
    }
    return {
      className: item.className,
      dist: Math.sqrt(sumSq),
    };
  });

  distances.sort((a, b) => a.dist - b.dist);
  const topK = distances.slice(0, Math.min(k, distances.length));

  const weightsByClass = new Map<string, number>();
  classes.forEach((c) => weightsByClass.set(c, 0));

  let totalWeight = 0;
  for (const neighbor of topK) {
    const weight = 1 / (neighbor.dist + 1e-4);
    weightsByClass.set(
      neighbor.className,
      (weightsByClass.get(neighbor.className) || 0) + weight
    );
    totalWeight += weight;
  }

  const colorMap = new Map(classMeta.map((c) => [c.name, c.color]));

  const probabilities = classes
    .map((className, idx) => ({
      className,
      probability:
        totalWeight > 0 ? (weightsByClass.get(className) || 0) / totalWeight : 0,
      color: colorMap.get(className) || getClassColor(idx),
    }))
    .sort((a, b) => b.probability - a.probability);

  return {
    className: probabilities[0].className,
    confidence: probabilities[0].probability,
    probabilities,
  };
}

/**
 * Evaluates Leave-One-Out (LOO) cross-validation accuracy for k-NN on the dataset.
 */
export function evaluateKNN(samples: CapturedSample[], k = 5): {
  accuracy: number;
  classes: string[];
} {
  const valid = samples
    .map((s) => ({ className: s.className, vec: extractFeatureVector(s.hands) }))
    .filter((s): s is { className: string; vec: number[] } => s.vec !== null);

  const classes = Array.from(new Set(valid.map((s) => s.className))).sort();
  if (classes.length < 2 || valid.length < 2) {
    return { accuracy: 0, classes };
  }

  let correct = 0;
  for (let i = 0; i < valid.length; i++) {
    const query = valid[i];
    const others = valid.filter((_, idx) => idx !== i);
    const dists = others.map((o) => {
      let sumSq = 0;
      for (let j = 0; j < query.vec.length; j++) {
        const d = query.vec[j] - o.vec[j];
        sumSq += d * d;
      }
      return { className: o.className, dist: Math.sqrt(sumSq) };
    });
    dists.sort((a, b) => a.dist - b.dist);
    const topK = dists.slice(0, Math.min(k, dists.length));
    const votes = new Map<string, number>();
    for (const n of topK) {
      const w = 1 / (n.dist + 1e-4);
      votes.set(n.className, (votes.get(n.className) || 0) + w);
    }
    let bestClass = "";
    let bestWeight = -1;
    votes.forEach((w, c) => {
      if (w > bestWeight) {
        bestWeight = w;
        bestClass = c;
      }
    });
    if (bestClass === query.className) correct++;
  }

  return {
    accuracy: Number(((correct / valid.length) * 100).toFixed(1)),
    classes,
  };
}
