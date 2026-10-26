"use client";

import React, { useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Download,
  HardDrive,
  Loader2,
  Play,
  Sliders,
  Sparkles,
} from "lucide-react";
import {
  CapturedSample,
  PredictionResult,
  TrainedModelRecord,
  TrainingConfig,
  TrainingHistoryPoint,
} from "@/types/handpose";

interface TrainerPanelProps {
  samples: CapturedSample[];
  config: TrainingConfig;
  onChangeConfig: (cfg: TrainingConfig) => void;
  isTraining: boolean;
  trainingProgress: TrainingHistoryPoint[];
  trainedModel: TrainedModelRecord | null;
  livePrediction: PredictionResult | null;
  onStartTraining: () => void;
  onExportModelZip: () => void;
  onSaveModelLocally: () => Promise<void>;
}

const CHART_W = 360;
const CHART_H = 110;

function TrainerPanel({
  samples,
  config,
  onChangeConfig,
  isTraining,
  trainingProgress,
  trainedModel,
  livePrediction,
  onStartTraining,
  onExportModelZip,
  onSaveModelLocally,
}: TrainerPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [savingLocal, setSavingLocal] = useState(false);
  const [savedNotice, setSavedNotice] = useState(false);

  // Check how many classes have at least 1 valid sample
  const classSampleCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const s of samples) {
      if (s.hands && s.hands.length > 0) {
        counts.set(
          s.className,
          (counts.get(s.className) || 0) + 1
        );
      }
    }
    return counts;
  }, [samples]);

  const populatedClasses = Array.from(classSampleCounts.entries());
  const canTrain = populatedClasses.length >= 2 && !isTraining;

  const historyToDisplay =
    isTraining || trainingProgress.length > 0
      ? trainingProgress
      : trainedModel?.history || [];

  const latestEpoch =
    historyToDisplay.length > 0
      ? historyToDisplay[historyToDisplay.length - 1]
      : null;

  const handleSaveLocal = async () => {
    try {
      setSavingLocal(true);
      await onSaveModelLocally();
      setSavedNotice(true);
      setTimeout(() => setSavedNotice(false), 2500);
    } finally {
      setSavingLocal(false);
    }
  };

  const chartW = CHART_W;
  const chartH = CHART_H;

  // Build SVG polyline points for Accuracy & Loss curves
  const { accPoints, lossPoints } = useMemo(() => {
    const maxEpochs = Math.max(config.epochs, historyToDisplay.length, 10);
    const maxLoss = Math.max(1.5, ...historyToDisplay.map((p) => p.loss));

    const acc = historyToDisplay
      .map((pt, i) => {
        const x =
          historyToDisplay.length === 1
            ? chartW / 2
            : (i / Math.max(1, maxEpochs - 1)) * (chartW - 16) + 8;
        const y = chartH - 10 - (pt.accuracy / 100) * (chartH - 20);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    const loss = historyToDisplay
      .map((pt, i) => {
        const x =
          historyToDisplay.length === 1
            ? chartW / 2
            : (i / Math.max(1, maxEpochs - 1)) * (chartW - 16) + 8;
        const y = chartH - 10 - Math.min(1, pt.loss / maxLoss) * (chartH - 20);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    return { accPoints: acc, lossPoints: loss };
  }, [historyToDisplay, config.epochs, chartW, chartH]);

  return (
    <div className="rounded-2xl border border-ocean-700/50 bg-[#101e2e]/90 p-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <BrainCircuit className="h-4 w-4 text-ocean-400" />
          <h2 className="text-sm font-bold text-white">
            2. Train Model &amp; Test Live on Webcam
          </h2>
        </div>

        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className={`flex items-center gap-1 rounded-lg border px-2.5 py-1 text-[11px] font-medium transition ${
            showAdvanced
              ? "border-gold-300/40 bg-gold-200/15 text-gold-200"
              : "border-ocean-700/60 bg-[#091522] text-slate-300 hover:text-white"
          }`}
        >
          <Sliders className="h-3 w-3" />
          <span>Advanced Options</span>
        </button>
      </div>

      {/* Classifier Engine Selector */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 mb-3">
        <button
          type="button"
          onClick={() => onChangeConfig({ ...config, classifierType: "mlp" })}
          className={`flex items-center gap-2.5 rounded-xl border p-2.5 text-left transition ${
            config.classifierType === "mlp"
              ? "border-ocean-500/80 bg-[#163554]/90 text-white shadow-sm"
              : "border-ocean-800/60 bg-[#0a1624]/70 text-slate-300 hover:border-ocean-700"
          }`}
        >
          <Cpu className="h-4 w-4 text-ocean-400 shrink-0" />
          <div>
            <div className="text-xs font-bold">TF.js Neural Network</div>
            <div className="text-[10px] text-sky-200/70">
              Recommended &bull; Trains in ~3s + downloadable model
            </div>
          </div>
        </button>

        <button
          type="button"
          onClick={() => onChangeConfig({ ...config, classifierType: "knn" })}
          className={`flex items-center gap-2.5 rounded-xl border p-2.5 text-left transition ${
            config.classifierType === "knn"
              ? "border-terracotta-500/70 bg-[#1c2e42] text-white shadow-sm"
              : "border-ocean-800/60 bg-[#0a1624]/70 text-slate-300 hover:border-ocean-700"
          }`}
        >
          <Sparkles className="h-4 w-4 text-gold-300 shrink-0" />
          <div>
            <div className="text-xs font-bold">Instant k-NN Matcher</div>
            <div className="text-[10px] text-sky-200/70">
              Zero training wait &bull; Nearest-neighbor matching
            </div>
          </div>
        </button>
      </div>

      {/* Hyperparameter Controls */}
      {showAdvanced && config.classifierType === "mlp" && (
        <div className="mb-3 grid grid-cols-2 gap-3 rounded-xl border border-ocean-700/60 bg-[#081320] p-3 text-xs sm:grid-cols-4">
          <div>
            <label className="block text-[11px] text-slate-400 mb-1">
              Epochs ({config.epochs})
            </label>
            <input
              type="range"
              min={15}
              max={120}
              step={5}
              value={config.epochs}
              onChange={(e) =>
                onChangeConfig({ ...config, epochs: Number(e.target.value) })
              }
              className="w-full accent-terracotta-500"
            />
          </div>

          <div>
            <label className="block text-[11px] text-slate-400 mb-1">
              Learning Rate
            </label>
            <select
              value={config.learningRate}
              onChange={(e) =>
                onChangeConfig({
                  ...config,
                  learningRate: Number(e.target.value),
                })
              }
              className="w-full rounded-lg border border-ocean-700/60 bg-[#0a1725] px-2 py-1 text-xs text-white"
            >
              <option value={0.001}>0.001 (Slow/Stable)</option>
              <option value={0.005}>0.005 (Balanced)</option>
              <option value={0.01}>0.010 (Fast)</option>
            </select>
          </div>

          <div>
            <label className="block text-[11px] text-slate-400 mb-1">
              Hidden Units
            </label>
            <select
              value={config.hiddenUnits}
              onChange={(e) =>
                onChangeConfig({
                  ...config,
                  hiddenUnits: Number(e.target.value),
                })
              }
              className="w-full rounded-lg border border-ocean-700/60 bg-[#0a1725] px-2 py-1 text-xs text-white"
            >
              <option value={32}>32 units</option>
              <option value={64}>64 units</option>
              <option value={128}>128 units</option>
            </select>
          </div>

          <div className="flex flex-col justify-center">
            <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-200">
              <input
                type="checkbox"
                checked={config.useAugmentation}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    useAugmentation: e.target.checked,
                  })
                }
                className="rounded accent-terracotta-500"
              />
              <span>5x 3D Augmentation</span>
            </label>
            <span className="text-[10px] text-slate-400">
              Rotates &amp; jitters landmarks
            </span>
          </div>
        </div>
      )}

      {/* Dataset Readiness Status & Train Action */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-ocean-700/60 bg-[#0c1825]/90 p-3">
        <div className="text-xs">
          {populatedClasses.length < 2 ? (
            <span className="text-gold-200">
              Collect samples for at least <strong>2 hand pose classes</strong> to
              enable training ({populatedClasses.length}/2 ready).
            </span>
          ) : (
            <span className="text-slate-200">
              Ready to train on{" "}
              <strong className="text-white">{samples.length} samples</strong>{" "}
              across{" "}
              <strong className="text-white">
                {populatedClasses.length} classes
              </strong>
              .
            </span>
          )}
        </div>

        <button
          onClick={onStartTraining}
          disabled={!canTrain}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-terracotta-500 to-terracotta-600 px-4 py-2 text-xs font-bold text-white shadow-md shadow-terracotta-950/40 transition hover:from-terracotta-400 hover:to-terracotta-500 disabled:cursor-not-allowed disabled:opacity-40 active:scale-95"
        >
          {isTraining ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>
                {latestEpoch
                  ? `Training Epoch ${latestEpoch.epoch}/${config.epochs}…`
                  : "Initializing Neural Network…"}
              </span>
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 fill-current" />
              <span>
                {config.classifierType === "mlp"
                  ? trainedModel
                    ? "Retrain Neural Network"
                    : "Train Neural Network"
                  : "Activate k-NN Classifier"}
              </span>
            </>
          )}
        </button>
      </div>

      {/* Live Training Curves & Model Telemetry */}
      {(historyToDisplay.length > 0 || trainedModel) && (
        <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
          {/* SVG Training Curves */}
          {historyToDisplay.length > 0 && (
            <div className="rounded-xl border border-ocean-700/60 bg-[#08121d] p-3">
              <div className="flex items-center justify-between text-[11px] mb-1.5">
                <span className="font-semibold text-slate-200">
                  Training Progress (Epoch {latestEpoch?.epoch}/{config.epochs})
                </span>
                <div className="flex items-center gap-3 font-mono">
                  <span className="text-emerald-400">
                    Acc: {latestEpoch?.accuracy ?? 0}%
                  </span>
                  <span className="text-rose-400">
                    Loss: {latestEpoch?.loss ?? 0}
                  </span>
                </div>
              </div>

              <svg
                viewBox={`0 0 ${chartW} ${chartH}`}
                className="h-28 w-full overflow-visible rounded-lg bg-[#0b1725]"
              >
                <line
                  x1="0"
                  y1={chartH * 0.25}
                  x2={chartW}
                  y2={chartH * 0.25}
                  stroke="#173552"
                  strokeDasharray="3 3"
                />
                <line
                  x1="0"
                  y1={chartH * 0.5}
                  x2={chartW}
                  y2={chartH * 0.5}
                  stroke="#173552"
                  strokeDasharray="3 3"
                />
                <line
                  x1="0"
                  y1={chartH * 0.75}
                  x2={chartW}
                  y2={chartH * 0.75}
                  stroke="#173552"
                  strokeDasharray="3 3"
                />

                {lossPoints && (
                  <polyline
                    fill="none"
                    stroke="#f43f5e"
                    strokeWidth="2"
                    points={lossPoints}
                  />
                )}

                {accPoints && (
                  <polyline
                    fill="none"
                    stroke="#34d399"
                    strokeWidth="2.5"
                    points={accPoints}
                  />
                )}
              </svg>
            </div>
          )}

          {/* Active Trained Model Summary & Export Actions */}
          {trainedModel && (
            <div className="flex flex-col justify-between rounded-xl border border-emerald-500/40 bg-[#0c241d]/40 p-3">
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-300">
                    <CheckCircle2 className="h-4 w-4" />
                    <span>{trainedModel.name}</span>
                  </div>
                  <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 font-mono text-[11px] font-bold text-emerald-300">
                    {trainedModel.finalAccuracy}% Accuracy
                  </span>
                </div>

                <div className="mt-2 flex flex-wrap gap-1">
                  {trainedModel.classes.map((cls) => (
                    <span
                      key={cls}
                      className="rounded-md border border-ocean-800 bg-[#091522] px-2 py-0.5 text-[10px] font-medium text-slate-200"
                    >
                      {cls}
                    </span>
                  ))}
                </div>

                {livePrediction && (
                  <div className="mt-2.5 flex items-center justify-between rounded-lg border border-ocean-800 bg-[#08121d] px-2.5 py-1.5 text-xs">
                    <span className="text-slate-400 flex items-center gap-1.5">
                      <Activity className="h-3.5 w-3.5 text-sky-400" />
                      Live Prediction:
                    </span>
                    <span className="font-bold text-white">
                      {livePrediction.className}{" "}
                      <span className="font-mono text-emerald-400">
                        ({(livePrediction.confidence * 100).toFixed(0)}%)
                      </span>
                    </span>
                  </div>
                )}
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                {trainedModel.serializedModel && (
                  <button
                    onClick={onExportModelZip}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-bold text-slate-950 transition hover:bg-emerald-400 shadow-sm"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download TF.js Model (.zip)</span>
                  </button>
                )}

                <button
                  onClick={handleSaveLocal}
                  disabled={savingLocal}
                  className="flex items-center justify-center gap-1.5 rounded-lg border border-ocean-700/60 bg-[#142c44] px-3 py-1.5 text-xs font-semibold text-slate-100 transition hover:bg-[#1b3a5a]"
                >
                  {savingLocal ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <HardDrive className="h-3.5 w-3.5 text-sky-400" />
                  )}
                  <span>
                    {savedNotice ? "Saved in Browser!" : "Save to Browser"}
                  </span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default React.memo(TrainerPanel);
