import * as tf from "@tensorflow/tfjs";
import {
  CapturedSample,
  DetectedHand,
  KNNExemplar,
  PoseClassMeta,
  PredictionResult,
  SerializedTFModel,
  TrainingConfig,
  TrainingHistoryPoint,
} from "@/types/handpose";
import {
  augmentFeatureVector,
  extractFeatureVector,
  FEATURE_VECTOR_SIZE,
  getClassColor,
} from "@/lib/landmarks";

export interface PreparedDataset {
  features: number[][];
  labels: number[];
  classes: string[];
  rawSamplesPerClass: Record<string, number>;
  valFeatures?: number[][];
  valLabels?: number[];
}

/**
 * Extracts normalized feature vectors from captured samples.
 * If validation samples are held out, data augmentation is applied
 * ONLY to training samples, preventing data leakage between train & validation splits.
 */
export function prepareTrainingDataset(
  samples: CapturedSample[],
  useAugmentation: boolean,
  augmentCopies = 4,
  holdoutValidation = true
): PreparedDataset {
  const validSamples = samples.filter(
    (s) => s.hands && s.hands.length > 0 && extractFeatureVector(s.hands) !== null
  );

  const classes = Array.from(new Set(validSamples.map((s) => s.className))).sort();
  const classToIndex = new Map<string, number>();
  classes.forEach((c, idx) => classToIndex.set(c, idx));

  const features: number[][] = [];
  const labels: number[] = [];
  const valFeatures: number[][] = [];
  const valLabels: number[] = [];
  const rawSamplesPerClass: Record<string, number> = {};

  // Group valid samples by class
  const samplesByClass = new Map<string, CapturedSample[]>();
  for (const c of classes) {
    samplesByClass.set(c, []);
  }
  for (const s of validSamples) {
    samplesByClass.get(s.className)?.push(s);
  }

  // Require at least 2 distinct classes with >= 4 samples before enabling holdout validation
  const classesWithEnoughSamples = Array.from(samplesByClass.values()).filter(
    (arr) => arr.length >= 4
  ).length;
  const shouldHoldout =
    holdoutValidation &&
    validSamples.length >= 12 &&
    classesWithEnoughSamples >= 2;

  for (const [className, classSamples] of samplesByClass.entries()) {
    const labelIdx = classToIndex.get(className);
    if (labelIdx === undefined) continue;

    rawSamplesPerClass[className] = classSamples.length;

    let trainClassSamples = classSamples;
    let valClassSamples: CapturedSample[] = [];

    if (shouldHoldout && classSamples.length >= 4) {
      const valCount = Math.max(1, Math.floor(classSamples.length * 0.15));
      trainClassSamples = classSamples.slice(0, classSamples.length - valCount);
      valClassSamples = classSamples.slice(classSamples.length - valCount);
    }

    // Process training samples (with augmentation if enabled)
    for (const sample of trainClassSamples) {
      const vec = extractFeatureVector(sample.hands);
      if (!vec) continue;

      features.push(vec);
      labels.push(labelIdx);

      if (useAugmentation) {
        for (let i = 0; i < augmentCopies; i++) {
          features.push(augmentFeatureVector(vec));
          labels.push(labelIdx);
        }
      }
    }

    // Process validation samples (strictly raw unaugmented vectors)
    for (const sample of valClassSamples) {
      const vec = extractFeatureVector(sample.hands);
      if (!vec) continue;

      valFeatures.push(vec);
      valLabels.push(labelIdx);
    }
  }

  return {
    features,
    labels,
    classes,
    rawSamplesPerClass,
    valFeatures: valFeatures.length > 0 ? valFeatures : undefined,
    valLabels: valLabels.length > 0 ? valLabels : undefined,
  };
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
 * Ensures TensorFlow.js is initialized with the optimal backend.
 * For lightweight tabular MLPs (63 input dimensions, ~6k parameters),
 * the CPU backend executes in < 0.6s for 40 epochs with ZERO WebGL shader compilation delay.
 * The WebGL backend incurs 1-2s of main-thread freeze during epoch 1 while compiling GLSL
 * shaders, and introduces GPU context contention with MediaPipe.
 */
export async function ensureTFBackend(): Promise<string> {
  if (typeof window !== "undefined") {
    try {
      if (tf.findBackend("cpu") && tf.getBackend() !== "cpu") {
        await tf.setBackend("cpu");
      }
    } catch {
      // Keep active backend
    }
  }
  await tf.ready();
  return tf.getBackend();
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
  await ensureTFBackend();

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
  const inputDim = prepared.features[0]?.length || FEATURE_VECTOR_SIZE;
  const xs = tf.tensor2d(shuffledFeatures, [shuffledFeatures.length, inputDim]);
  const labelsTensor = tf.tensor1d(shuffledLabels, "int32");
  const ys = tf.oneHot(labelsTensor, numClasses);
  labelsTensor.dispose();

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [inputDim],
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

  const distinctValClasses = new Set(prepared.valLabels || []).size;
  const hasValidationData = Boolean(
    prepared.valFeatures &&
    prepared.valFeatures.length > 0 &&
    prepared.valLabels &&
    prepared.valLabels.length > 0 &&
    distinctValClasses >= 2
  );

  let valXs: tf.Tensor2D | null = null;
  let valYs: tf.Tensor | null = null;

  if (hasValidationData && prepared.valFeatures && prepared.valLabels) {
    valXs = tf.tensor2d(prepared.valFeatures, [
      prepared.valFeatures.length,
      inputDim,
    ]);
    const valLabelsTensor = tf.tensor1d(prepared.valLabels, "int32");
    valYs = tf.oneHot(valLabelsTensor, numClasses);
    valLabelsTensor.dispose();
  }

  // Yield one animation frame so the UI paints the initial training state before computation begins
  await tf.nextFrame();

  let trainedSuccessfully = false;
  try {
    await model.fit(xs, ys, {
      epochs: config.epochs,
      batchSize: config.batchSize,
      validationData: valXs && valYs ? [valXs, valYs] : undefined,
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

    const serializedModel = await serializeTFModel(model);
    trainedSuccessfully = true;

    return {
      model,
      classes,
      history,
      serializedModel,
    };
  } finally {
    xs.dispose();
    ys.dispose();
    valXs?.dispose();
    valYs?.dispose();
    if (!trainedSuccessfully) {
      model.dispose();
    }
  }
}

/**
 * High-performance chunked conversion from Uint8Array to Base64
 * avoiding string allocation memory churn on large typed arrays.
 */
export function uint8ArrayToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk as unknown as number[]);
  }
  return btoa(binary);
}

/**
 * Decodes a base64 string directly into a Uint8Array buffer.
 */
export function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
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
      const weightDataBase64 = uint8ArrayToBase64(bytes);

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
  await ensureTFBackend();
  const bytes = base64ToUint8Array(serialized.weightDataBase64);

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

  // Adaptively support both 131-D models and legacy 63-D models
  const expectedDim = (model.inputs[0]?.shape?.[1] as number) || FEATURE_VECTOR_SIZE;
  let inputVec = vec;
  if (inputVec.length < expectedDim) {
    inputVec = [...inputVec, ...new Array(expectedDim - inputVec.length).fill(0)];
  } else if (inputVec.length > expectedDim) {
    inputVec = inputVec.slice(0, expectedDim);
  }

  const probs = tf.tidy(() => {
    const input = tf.tensor2d([inputVec], [1, expectedDim]);
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
 * Instant k-Nearest Neighbors (k-NN) classifier over normalized landmark vectors.
 * Uses distance-weighted voting over the top k=5 neighbors.
 */
export function predictWithKNN(
  exemplars: KNNExemplar[],
  hands: DetectedHand[],
  classMeta: PoseClassMeta[],
  k = 5
): PredictionResult | null {
  const queryVec = extractFeatureVector(hands);
  if (!queryVec || exemplars.length === 0) return null;

  const validSamples = exemplars;
  const classes = Array.from(new Set(validSamples.map((s) => s.className))).sort();
  if (classes.length < 2) return null;

  const distances = validSamples.map((item) => {
    let sumSq = 0;
    const maxDim = Math.max(queryVec.length, item.vec.length, FEATURE_VECTOR_SIZE);
    for (let i = 0; i < maxDim; i++) {
      const q = queryVec[i] ?? 0;
      const v = item.vec[i] ?? 0;
      const diff = q - v;
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
export function evaluateKNN(exemplars: KNNExemplar[], k = 5): {
  accuracy: number;
  classes: string[];
} {
  const classes = Array.from(new Set(exemplars.map((s) => s.className))).sort();
  if (classes.length < 2 || exemplars.length < 2) {
    return { accuracy: 0, classes };
  }

  let correct = 0;
  for (let i = 0; i < exemplars.length; i++) {
    const query = exemplars[i];
    const dists: { className: string; dist: number }[] = [];

    for (let j = 0; j < exemplars.length; j++) {
      if (j === i) continue;
      const other = exemplars[j];
      let sumSq = 0;
      const maxDim = Math.max(query.vec.length, other.vec.length, FEATURE_VECTOR_SIZE);
      for (let d = 0; d < maxDim; d++) {
        const q = query.vec[d] ?? 0;
        const o = other.vec[d] ?? 0;
        const diff = q - o;
        sumSq += diff * diff;
      }
      dists.push({ className: other.className, dist: Math.sqrt(sumSq) });
    }

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
    accuracy: Number(((correct / exemplars.length) * 100).toFixed(1)),
    classes,
  };
}
