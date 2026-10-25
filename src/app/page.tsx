"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Navbar from "@/components/Navbar";
import CameraStage from "@/components/CameraStage";
import ClassManager from "@/components/ClassManager";
import TrainerPanel from "@/components/TrainerPanel";
import DatasetHubModal from "@/components/DatasetHubModal";
import TutorialModal from "@/components/TutorialModal";
import {
  CapturedSample,
  DetectedHand,
  PoseClassMeta,
  PredictionResult,
  TrainedModelRecord,
  TrainingConfig,
  TrainingHistoryPoint,
} from "@/types/handpose";
import { getClassColor } from "@/lib/landmarks";
import {
  deserializeTFModel,
  evaluateKNN,
  predictWithKNN,
  predictWithMLP,
  trainMLPClassifier,
} from "@/lib/classifier";
import {
  loadPreferencesFromLocalStorage,
  loadWorkspaceLocally,
  saveModelLocally,
  savePreferencesToLocalStorage,
  saveWorkspaceLocally,
} from "@/lib/storage";
import { exportDatasetZip, exportTrainedModelZip } from "@/lib/export";
import { generateDemoDataset } from "@/lib/demoData";
import {
  ArrowRight,
  BookOpen,
  BrainCircuit,
  FolderPlus,
  Layers,
  Sparkles,
  X,
} from "lucide-react";

const INITIAL_CLASSES: PoseClassMeta[] = [
  { name: "Thumbs Up", color: getClassColor(0), createdAt: 1 },
  { name: "Peace", color: getClassColor(1), createdAt: 2 },
];

export default function HandPoseStudioPage() {
  const [classes, setClasses] = useState<PoseClassMeta[]>(INITIAL_CLASSES);
  const [samples, setSamples] = useState<CapturedSample[]>([]);
  const [activeClassName, setActiveClassName] = useState<string | null>(
    INITIAL_CLASSES[0].name
  );

  // Step-by-step workspace view to prevent overwhelming the user
  const [activeStepTab, setActiveStepTab] = useState<
    "collect" | "train" | "all"
  >("collect");

  // MediaPipe & Telemetry State
  const [modelReady, setModelReady] = useState(false);
  const [fps, setFps] = useState(0);
  const lastFpsUpdateRef = useRef<number>(0);
  const lastPredictionTimeRef = useRef<number>(0);
  const [syncStatus, setSyncStatus] = useState<"saved" | "saving" | "idle">(
    "saved"
  );

  // Classifier & Training State
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    classifierType: "mlp",
    epochs: 45,
    learningRate: 0.005,
    batchSize: 16,
    hiddenUnits: 64,
    useAugmentation: true,
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<
    TrainingHistoryPoint[]
  >([]);
  const [trainedModelRecord, setTrainedModelRecord] =
    useState<TrainedModelRecord | null>(null);
  const tfModelRef = useRef<tf.LayersModel | null>(null);
  const [livePrediction, setLivePrediction] =
    useState<PredictionResult | null>(null);

  // Modals & Help
  const [hubOpen, setHubOpen] = useState(false);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [showHelp, setShowHelp] = useState(true);
  const [hydrated, setHydrated] = useState(false);

  // Load persisted workspace & preferences from browser storage on mount
  useEffect(() => {
    let mounted = true;
    async function hydrate() {
      const prefs = loadPreferencesFromLocalStorage();
      if (!prefs.hasCompletedTutorial) {
        setTutorialOpen(true);
      }
      setShowHelp(!prefs.hideQuickGuide);

      const saved = await loadWorkspaceLocally();
      if (!mounted) return;

      if (saved && (saved.classes.length > 0 || saved.samples.length > 0)) {
        setClasses(saved.classes);
        setSamples(saved.samples);
        setActiveClassName(saved.classes[0]?.name || null);

        if (saved.activeModel) {
          setTrainedModelRecord(saved.activeModel);
          if (saved.activeModel.serializedModel) {
            try {
              const restored = await deserializeTFModel(
                saved.activeModel.serializedModel
              );
              if (mounted) tfModelRef.current = restored;
            } catch (err) {
              console.warn(
                "Could not restore TF.js model from local storage:",
                err
              );
            }
          }
        }
      }
      setHydrated(true);
    }
    hydrate();
    return () => {
      mounted = false;
    };
  }, []);

  // Auto-save workspace strictly to browser local storage whenever classes, samples, or model change
  useEffect(() => {
    if (!hydrated) return;
    setSyncStatus("saving");
    const timer = setTimeout(async () => {
      await saveWorkspaceLocally(classes, samples, trainedModelRecord);
      setSyncStatus("saved");
    }, 350);
    return () => clearTimeout(timer);
  }, [classes, samples, trainedModelRecord, hydrated]);

  const activeClass =
    classes.find((c) => c.name === activeClassName) || classes[0] || null;

  // Class & Sample handlers
  const handleAddClass = (name: string) => {
    const exists = classes.some(
      (c) => c.name.toLowerCase() === name.toLowerCase()
    );
    if (exists) {
      setActiveClassName(
        classes.find((c) => c.name.toLowerCase() === name.toLowerCase())!.name
      );
      return;
    }
    const newCls: PoseClassMeta = {
      name,
      color: getClassColor(classes.length),
      createdAt: Date.now(),
    };
    setClasses((prev) => [...prev, newCls]);
    setActiveClassName(name);
  };

  const handleRenameClass = (oldName: string, newName: string) => {
    setClasses((prev) =>
      prev.map((c) => (c.name === oldName ? { ...c, name: newName } : c))
    );
    setSamples((prev) =>
      prev.map((s) =>
        s.className === oldName ? { ...s, className: newName } : s
      )
    );
    if (activeClassName === oldName) {
      setActiveClassName(newName);
    }
  };

  const handleDeleteClass = (name: string) => {
    setClasses((prev) => {
      const next = prev.filter((c) => c.name !== name);
      if (activeClassName === name) {
        setActiveClassName(next[0]?.name || null);
      }
      return next;
    });
    setSamples((prev) => prev.filter((s) => s.className !== name));
  };

  const handleCaptureSample = useCallback((sample: CapturedSample) => {
    setSamples((prev) => [...prev, sample]);
  }, []);

  const handleDeleteSample = (id: string) => {
    setSamples((prev) => prev.filter((s) => s.id !== id));
  };

  const handleClearWorkspace = () => {
    setSamples([]);
    setTrainedModelRecord(null);
    setTrainingProgress([]);
    setLivePrediction(null);
    if (tfModelRef.current) {
      tfModelRef.current.dispose();
      tfModelRef.current = null;
    }
  };

  const handleLoadDemoDataset = () => {
    const demo = generateDemoDataset(8);
    setClasses(demo.classes);
    setSamples(demo.samples);
    setActiveClassName(demo.classes[0].name);
  };

  // Real-time hand detection callback + live gesture classification (throttled to avoid React re-render cascades)
  const handleHandsDetected = useCallback(
    (detectedHands: DetectedHand[], currentFps: number) => {
      const now = performance.now();
      if (now - lastFpsUpdateRef.current >= 800) {
        lastFpsUpdateRef.current = now;
        setFps(currentFps);
      }

      if (!detectedHands || detectedHands.length === 0) {
        setLivePrediction(null);
        return;
      }

      if (now - lastPredictionTimeRef.current < 70) {
        return;
      }
      lastPredictionTimeRef.current = now;

      if (
        trainedModelRecord?.classifierType === "mlp" &&
        tfModelRef.current
      ) {
        const pred = predictWithMLP(
          tfModelRef.current,
          trainedModelRecord.classes,
          detectedHands,
          classes
        );
        setLivePrediction(pred);
      } else if (
        trainedModelRecord?.classifierType === "knn" ||
        (trainingConfig.classifierType === "knn" && samples.length >= 4)
      ) {
        const pred = predictWithKNN(samples, detectedHands, classes, 5);
        setLivePrediction(pred);
      } else {
        setLivePrediction(null);
      }
    },
    [trainedModelRecord, trainingConfig.classifierType, samples, classes]
  );

  // Train classifier in browser
  const handleStartTraining = async () => {
    if (isTraining) return;

    if (trainingConfig.classifierType === "knn") {
      const evalResult = evaluateKNN(samples, 5);
      const record: TrainedModelRecord = {
        id: `knn_${Date.now()}`,
        name: `k-NN Pose Classifier (${evalResult.classes.length} classes)`,
        classifierType: "knn",
        classes: evalResult.classes,
        sampleCount: samples.length,
        finalAccuracy: evalResult.accuracy,
        finalLoss: 0,
        epochs: 1,
        trainedAt: Date.now(),
        history: [],
      };
      setTrainedModelRecord(record);
      return;
    }

    try {
      setIsTraining(true);
      setTrainingProgress([]);

      const result = await trainMLPClassifier(
        samples,
        trainingConfig,
        (point) => {
          setTrainingProgress((prev) => [...prev, point]);
        }
      );

      if (tfModelRef.current) {
        tfModelRef.current.dispose();
      }
      tfModelRef.current = result.model;

      const lastPt = result.history[result.history.length - 1];
      const record: TrainedModelRecord = {
        id: `mlp_${Date.now()}`,
        name: `TF.js MLP (${result.classes.length} classes)`,
        classifierType: "mlp",
        classes: result.classes,
        sampleCount: samples.length,
        finalAccuracy: lastPt?.accuracy ?? 100,
        finalLoss: lastPt?.loss ?? 0,
        epochs: trainingConfig.epochs,
        trainedAt: Date.now(),
        history: result.history,
        serializedModel: result.serializedModel,
      };

      setTrainedModelRecord(record);
    } catch (err) {
      console.error("Training error:", err);
      alert(
        err instanceof Error ? err.message : "Failed to train neural network"
      );
    } finally {
      setIsTraining(false);
    }
  };

  const handleRestoreModel = async (model: TrainedModelRecord) => {
    setTrainedModelRecord(model);
    if (model.serializedModel) {
      try {
        const restored = await deserializeTFModel(model.serializedModel);
        if (tfModelRef.current) tfModelRef.current.dispose();
        tfModelRef.current = restored;
      } catch (err) {
        console.error("Failed to deserialize model:", err);
      }
    }
  };

  const handleToggleQuickGuide = () => {
    setShowHelp((prev) => {
      const next = !prev;
      savePreferencesToLocalStorage({ hideQuickGuide: !next });
      return next;
    });
  };

  const handleCloseTutorial = () => {
    setTutorialOpen(false);
    savePreferencesToLocalStorage({ hasCompletedTutorial: true });
  };

  // Count how many classes have at least 1 sample
  const readyClassCount = new Set(samples.map((s) => s.className)).size;

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <Navbar
        modelReady={modelReady}
        fps={fps}
        totalSamples={samples.length}
        totalClasses={classes.length}
        hasTrainedModel={Boolean(trainedModelRecord)}
        trainedAccuracy={trainedModelRecord?.finalAccuracy}
        syncStatus={syncStatus}
        onOpenHub={() => setHubOpen(true)}
        onQuickDownloadZip={() =>
          exportDatasetZip(classes, samples, "gesture-dataset")
        }
        onLoadDemoData={handleLoadDemoDataset}
        onOpenTutorial={() => setTutorialOpen(true)}
        onToggleHelp={handleToggleQuickGuide}
      />

      {/* Quick Studio Guide Banner (Stacked vertically on separate lines!) */}
      {showHelp && (
        <div className="border-b border-cyan-500/30 bg-slate-900/95 px-4 py-3.5 sm:px-6">
          <div className="mx-auto flex max-w-[1600px] items-start justify-between gap-4">
            <div className="flex flex-col gap-2 text-xs text-slate-300">
              <div className="flex items-center gap-2 font-bold text-cyan-300 text-sm">
                <BookOpen className="h-4 w-4 shrink-0" />
                <span>Quick Studio Guide:</span>
              </div>
              <ol className="flex flex-col space-y-1.5 list-decimal list-inside text-slate-200 pl-1">
                <li>
                  <strong>Select or add a gesture class</strong> in Step 1 (or
                  click <em>Demo Data</em> in the top bar to load 32 ready-made
                  samples).
                </li>
                <li>
                  <strong>Capture photos</strong> by holding your hand in front
                  of the camera and pressing{" "}
                  <kbd className="rounded border border-slate-700 bg-slate-800 px-1.5 py-0.5 font-mono text-[10px]">
                    Space
                  </kbd>{" "}
                  or <strong>Auto-Burst</strong>.
                </li>
                <li>
                  <strong>Train your classifier</strong> in Step 2 to recognize
                  your hand gestures live on the webcam feed.
                </li>
                <li>
                  <strong>Download your dataset or model</strong> as a{" "}
                  <strong>.zip</strong>, <strong>.csv</strong>, or{" "}
                  <strong>TF.js bundle</strong> (all work is saved privately in
                  your browser&apos;s local storage).
                </li>
              </ol>
            </div>
            <button
              onClick={handleToggleQuickGuide}
              title="Dismiss Quick Studio Guide"
              className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Main Studio Grid */}
      <main className="mx-auto grid w-full max-w-[1600px] flex-1 grid-cols-1 gap-6 p-4 sm:p-6 lg:grid-cols-12">
        {/* Left Column: Live Camera + MediaPipe 3D Stage (5 cols on lg) */}
        <div className="flex flex-col gap-4 lg:col-span-5">
          <CameraStage
            activeClass={activeClass}
            onCaptureSample={handleCaptureSample}
            onHandsDetected={handleHandsDetected}
            onModelStatusChange={setModelReady}
            livePrediction={livePrediction}
            classifierActive={Boolean(trainedModelRecord)}
          />

          {/* Helpful Starter Callout when 0 samples exist */}
          {samples.length === 0 && (
            <div className="flex items-center justify-between gap-3 rounded-2xl border border-violet-500/30 bg-violet-500/10 p-3.5 text-xs">
              <div className="flex items-center gap-2.5">
                <Sparkles className="h-4 w-4 text-violet-300 shrink-0" />
                <span className="text-slate-200">
                  No webcam handy? Load 32 synthetic 3D hand poses to try
                  training right away.
                </span>
              </div>
              <button
                onClick={handleLoadDemoDataset}
                className="shrink-0 rounded-lg bg-violet-500 px-3 py-1.5 font-bold text-slate-950 transition hover:bg-violet-400"
              >
                Load Demo Data
              </button>
            </div>
          )}
        </div>

        {/* Right Column: Guided Step Tabs (7 cols on lg) */}
        <div className="flex flex-col gap-4 lg:col-span-7">
          {/* Step Switcher Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-slate-800 bg-slate-900/90 p-2">
            <div className="flex flex-1 items-center gap-1.5">
              <button
                onClick={() => setActiveStepTab("collect")}
                className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                  activeStepTab === "collect"
                    ? "bg-cyan-500 text-slate-950 shadow-sm"
                    : "text-slate-300 hover:bg-slate-800"
                }`}
              >
                <FolderPlus className="h-4 w-4" />
                <span>Step 1: Collect Poses ({samples.length})</span>
              </button>

              <button
                onClick={() => setActiveStepTab("train")}
                className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                  activeStepTab === "train"
                    ? "bg-violet-500 text-white shadow-sm"
                    : "text-slate-300 hover:bg-slate-800"
                }`}
              >
                <BrainCircuit className="h-4 w-4" />
                <span>
                  Step 2: Train &amp; Test Live
                  {trainedModelRecord
                    ? ` (${trainedModelRecord.finalAccuracy}%)`
                    : ""}
                </span>
              </button>
            </div>

            <button
              onClick={() =>
                setActiveStepTab((prev) =>
                  prev === "all" ? "collect" : "all"
                )
              }
              title="Show both steps stacked together"
              className={`flex items-center gap-1 rounded-xl border px-3 py-2 text-xs font-medium transition ${
                activeStepTab === "all"
                  ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-300"
                  : "border-slate-800 bg-slate-950 text-slate-400 hover:text-white"
              }`}
            >
              <Layers className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">
                {activeStepTab === "all" ? "Tabbed View" : "Show All"}
              </span>
            </button>
          </div>

          {/* Step 1 Panel: Collect Poses */}
          {(activeStepTab === "collect" || activeStepTab === "all") && (
            <>
              <ClassManager
                classes={classes}
                samples={samples}
                activeClassName={activeClass?.name || null}
                onSelectClass={setActiveClassName}
                onAddClass={handleAddClass}
                onRenameClass={handleRenameClass}
                onDeleteClass={handleDeleteClass}
                onDeleteSample={handleDeleteSample}
                onClearAll={handleClearWorkspace}
              />

              {/* Next Step Callout Banner when user has collected >=2 classes in Tabbed mode */}
              {activeStepTab === "collect" && readyClassCount >= 2 && (
                <div className="flex items-center justify-between gap-3 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-3.5 text-xs">
                  <span className="text-emerald-200">
                    Awesome! You have samples for{" "}
                    <strong>{readyClassCount} classes</strong> ({samples.length}{" "}
                    total photos). Ready to train your gesture classifier?
                  </span>
                  <button
                    onClick={() => setActiveStepTab("train")}
                    className="flex shrink-0 items-center gap-1.5 rounded-xl bg-emerald-500 px-4 py-2 font-bold text-slate-950 transition hover:bg-emerald-400"
                  >
                    <span>Go to Step 2: Train Model</span>
                    <ArrowRight className="h-4 w-4" />
                  </button>
                </div>
              )}
            </>
          )}

          {/* Step 2 Panel: Train & Test Live */}
          {(activeStepTab === "train" || activeStepTab === "all") && (
            <TrainerPanel
              samples={samples}
              config={trainingConfig}
              onChangeConfig={setTrainingConfig}
              isTraining={isTraining}
              trainingProgress={trainingProgress}
              trainedModel={trainedModelRecord}
              livePrediction={livePrediction}
              onStartTraining={handleStartTraining}
              onExportModelZip={() => {
                if (trainedModelRecord) {
                  exportTrainedModelZip(trainedModelRecord);
                }
              }}
              onSaveModelLocally={async () => {
                if (trainedModelRecord) {
                  await saveModelLocally(trainedModelRecord);
                }
              }}
            />
          )}
        </div>
      </main>

      {/* Interactive Onboarding Walkthrough Modal */}
      <TutorialModal
        isOpen={tutorialOpen}
        onClose={handleCloseTutorial}
        onLoadDemoAndClose={() => {
          handleLoadDemoDataset();
          handleCloseTutorial();
        }}
      />

      {/* Datasets, Local Browser Snapshots & Export Hub Modal */}
      <DatasetHubModal
        isOpen={hubOpen}
        onClose={() => setHubOpen(false)}
        classes={classes}
        samples={samples}
        activeModel={trainedModelRecord}
        onRestoreWorkspace={(newClasses, newSamples) => {
          setClasses(newClasses);
          setSamples(newSamples);
          setActiveClassName(newClasses[0]?.name || null);
        }}
        onRestoreModel={handleRestoreModel}
      />
    </div>
  );
}
