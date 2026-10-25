"use client";

import React, { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BrainCircuit,
  Camera,
  CheckCircle2,
  Download,
  FolderPlus,
  HardDrive,
  Sparkles,
  X,
} from "lucide-react";

interface TutorialModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoadDemoAndClose: () => void;
}

const TUTORIAL_STEPS = [
  {
    stepNumber: 1,
    title: "Create Your Hand Pose Classes",
    subtitle: "Define the hand poses you want to recognize",
    icon: FolderPlus,
    accent: "from-cyan-500 to-blue-600",
    description:
      "Start by choosing at least 2 hand pose labels (like 'Thumbs Up' and 'Peace'). We've already added two starter classes for you, or you can click any of the quick preset chips to add more.",
    tips: [
      "Click a class card to make it the active recording target.",
      "You can rename or delete classes at any time.",
      "No webcam handy? Click 'Try with Demo Dataset' below to load 32 ready-made 3D hand samples!",
    ],
  },
  {
    stepNumber: 2,
    title: "Capture Webcam Samples",
    subtitle: "Record 10–15 examples per hand pose",
    icon: Camera,
    accent: "from-amber-500 to-orange-600",
    description:
      "Hold your hand in front of your webcam. MediaPipe automatically tracks 21 3D hand joints in real time. Capture samples from slightly different angles and distances for the best model accuracy.",
    tips: [
      "Press the Spacebar or click 'Capture Photo' for single shots.",
      "Use 'Auto-Burst (15)' to automatically record 15 frames in 3 seconds while holding a pose.",
      "Use the '3s Timer' if you need a moment to get your hand into position.",
    ],
  },
  {
    stepNumber: 3,
    title: "Train & Test Live in Your Browser",
    subtitle: "Zero server setup — powered by TensorFlow.js",
    icon: BrainCircuit,
    accent: "from-violet-500 to-indigo-600",
    description:
      "Switch to Step 2 ('Train & Test Live') and click 'Train Neural Network'. In just a few seconds, a custom classifier trains on your 3D hand landmarks and starts predicting your hand poses live on the webcam feed!",
    tips: [
      "We normalize hand positions relative to your wrist so distance from the camera won't confuse the model.",
      "You can also switch to 'Instant k-NN' for zero-wait nearest-neighbor matching.",
    ],
  },
  {
    stepNumber: 4,
    title: "100% Local Browser Storage & Export",
    subtitle: "Your photos & models never leave your device",
    icon: Download,
    accent: "from-emerald-500 to-teal-600",
    description:
      "All captured images, 3D landmarks, and trained models are saved strictly in your browser's local storage — nothing is ever uploaded to a server. When you're ready, download your dataset or trained model with one click.",
    tips: [
      "Download ZIP: Organized folders with .jpeg images + .json annotations + landmarks.csv.",
      "Export CSV: Flat table of 21 3D landmarks ready for Python, scikit-learn, or PyTorch.",
      "Export TF.js Model: Download model.json + weights.bin to use in your own web apps.",
    ],
  },
];

export default function TutorialModal({
  isOpen,
  onClose,
  onLoadDemoAndClose,
}: TutorialModalProps) {
  const [currentIdx, setCurrentIdx] = useState(0);

  if (!isOpen) return null;

  const step = TUTORIAL_STEPS[currentIdx];
  const IconComponent = step.icon;
  const isLast = currentIdx === TUTORIAL_STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4 backdrop-blur-sm">
      <div className="relative flex w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-ocean-700/60 bg-[#0e1d2f] shadow-2xl max-h-[90vh] overflow-y-auto">
        {/* Top Progress Bar */}
        <div className="grid grid-cols-4 gap-1 bg-[#08121d] p-1.5">
          {TUTORIAL_STEPS.map((s, idx) => (
            <button
              key={s.stepNumber}
              onClick={() => setCurrentIdx(idx)}
              className={`h-1.5 rounded-full transition-all ${
                idx <= currentIdx ? "bg-terracotta-500" : "bg-[#14263a]"
              }`}
            />
          ))}
        </div>

        {/* Header */}
        <div className="flex items-center justify-between px-4 sm:px-6 pt-4 pb-2">
          <span className="rounded-full border border-gold-300/40 bg-gold-200/15 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-gold-200">
            Step {step.stepNumber} of {TUTORIAL_STEPS.length}
          </span>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-[#152e4a] hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="px-4 sm:px-6 py-3 sm:py-4">
          <div className="flex items-center gap-3.5 mb-3.5">
            <div
              className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br ${step.accent} text-white shadow-lg`}
            >
              <IconComponent className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-extrabold text-white">
                {step.title}
              </h2>
              <p className="text-xs text-sky-200/70">{step.subtitle}</p>
            </div>
          </div>

          <p className="text-xs leading-relaxed text-slate-200">
            {step.description}
          </p>

          <div className="mt-4 rounded-xl border border-ocean-700/60 bg-[#081320] p-3 sm:p-3.5">
            <div className="text-[11px] font-bold uppercase tracking-wider text-gold-200 mb-2">
              Helpful Tips
            </div>
            <ul className="space-y-2">
              {step.tips.map((tip, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-xs text-slate-200"
                >
                  <CheckCircle2 className="h-3.5 w-3.5 text-terracotta-400 shrink-0 mt-0.5" />
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Privacy Guarantee Badge */}
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-950/20 px-3 py-2 text-[11px] text-emerald-200">
            <HardDrive className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
            <span>
              <strong>100% Browser Local Storage:</strong> All photos and models
              stay privately on your device.
            </span>
          </div>
        </div>

        {/* Footer Navigation */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-ocean-700/50 bg-[#0a1624] px-4 sm:px-6 py-3 sm:py-4">
          <button
            onClick={onLoadDemoAndClose}
            className="flex items-center gap-1.5 rounded-lg border border-ocean-600/60 bg-[#163554] px-3 py-1.5 text-xs font-semibold text-sky-100 transition hover:bg-[#1f456c]"
          >
            <Sparkles className="h-3.5 w-3.5 text-gold-300" />
            <span className="text-[11px] sm:text-xs">Try with Demo Dataset</span>
          </button>

          <div className="flex items-center gap-2">
            {currentIdx > 0 && (
              <button
                onClick={() => setCurrentIdx((i) => i - 1)}
                className="flex items-center gap-1 rounded-lg border border-ocean-700/60 bg-[#13273e] px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-[#1a3857]"
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                <span>Back</span>
              </button>
            )}

            {!isLast ? (
              <button
                onClick={() => setCurrentIdx((i) => i + 1)}
                className="flex items-center gap-1 rounded-lg bg-gradient-to-r from-terracotta-500 to-terracotta-600 px-4 py-1.5 text-xs font-bold text-white hover:from-terracotta-400 hover:to-terracotta-500 shadow-sm shadow-terracotta-950/40"
              >
                <span>Next</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            ) : (
              <button
                onClick={onClose}
                className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-terracotta-500 to-terracotta-600 px-4 py-1.5 text-xs font-bold text-white hover:from-terracotta-400 hover:to-terracotta-500 shadow-md shadow-terracotta-950/40"
              >
                <span>Start Using Studio</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
