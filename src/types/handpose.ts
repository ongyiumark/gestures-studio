export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface DetectedHand {
  handedness: "Left" | "Right";
  score: number;
  landmarks: Point3D[];
  worldLandmarks: Point3D[];
}

export interface CapturedSample {
  id: string;
  className: string;
  timestamp: number;
  imageDataUrl: string;
  width: number;
  height: number;
  hands: DetectedHand[];
}

export interface PoseClassMeta {
  name: string;
  color: string;
  createdAt: number;
}

export type ClassifierType = "mlp" | "knn";

export interface TrainingConfig {
  classifierType: ClassifierType;
  epochs: number;
  learningRate: number;
  batchSize: number;
  hiddenUnits: number;
  useAugmentation: boolean;
}

export interface TrainingHistoryPoint {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss?: number;
  valAccuracy?: number;
}

export interface SerializedTFModel {
  modelTopology: unknown;
  weightSpecs: unknown;
  weightDataBase64: string;
}

export interface KNNExemplar {
  className: string;
  vec: number[];
}

export interface TrainedModelRecord {
  id: string;
  name: string;
  classifierType: ClassifierType;
  classes: string[];
  sampleCount: number;
  finalAccuracy: number;
  finalLoss: number;
  epochs: number;
  trainedAt: number;
  history: TrainingHistoryPoint[];
  serializedModel?: SerializedTFModel;
  knnExemplars?: KNNExemplar[];
}

export interface PredictionResult {
  className: string;
  confidence: number;
  probabilities: {
    className: string;
    probability: number;
    color: string;
  }[];
}

export interface DatasetSnapshot {
  id: string;
  name: string;
  description: string;
  createdAt: number;
  updatedAt: number;
  classes: PoseClassMeta[];
  samples: CapturedSample[];
}

export interface DatasetSnapshotSummary {
  id: string;
  name: string;
  description: string;
  createdAt: number;
  updatedAt: number;
  sampleCount: number;
  classes: PoseClassMeta[];
}
