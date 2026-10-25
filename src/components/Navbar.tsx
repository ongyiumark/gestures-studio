"use client";

import React from "react";
import {
  Activity,
  BookOpen,
  Download,
  FolderArchive,
  Hand,
  HardDrive,
  HelpCircle,
  Loader2,
  Sparkles,
} from "lucide-react";

interface NavbarProps {
  modelReady: boolean;
  fps: number;
  totalSamples: number;
  totalClasses: number;
  hasTrainedModel: boolean;
  trainedAccuracy?: number;
  syncStatus: "saved" | "saving" | "idle";
  onOpenHub: () => void;
  onQuickDownloadZip: () => void;
  onLoadDemoData: () => void;
  onOpenTutorial: () => void;
  onToggleHelp: () => void;
}

export default function Navbar({
  modelReady,
  fps,
  totalSamples,
  totalClasses,
  hasTrainedModel,
  trainedAccuracy,
  syncStatus,
  onOpenHub,
  onQuickDownloadZip,
  onLoadDemoData,
  onOpenTutorial,
  onToggleHelp,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
        {/* Brand Identity */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-indigo-600 shadow-lg shadow-cyan-500/20">
            <Hand className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-white sm:text-lg">
                Gesture Studio
              </h1>
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-cyan-300">
                v2.0
              </span>
            </div>
            <p className="text-xs text-slate-400">
              3D Hand Pose Dataset Builder &amp; In-Browser Classifier &bull; by{" "}
              <a
                href="https://github.com/ongyiumark/"
                target="_blank"
                rel="noreferrer"
                className="font-medium text-cyan-400 hover:text-cyan-300 hover:underline"
              >
                ongyiumark
              </a>
            </p>
          </div>
        </div>

        {/* Live Telemetry Badges */}
        <div className="hidden items-center gap-2 xl:flex">
          <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/90 px-3 py-1.5 text-xs">
            {modelReady ? (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
                <span className="font-medium text-slate-200">
                  Tracker Ready
                </span>
                <span className="text-slate-500">|</span>
                <span className="font-mono text-cyan-400">{fps} FPS</span>
              </>
            ) : (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-400" />
                <span className="text-amber-300">Loading 3D Tracker…</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2.5 rounded-lg border border-slate-800 bg-slate-900/90 px-3 py-1.5 text-xs text-slate-300">
            <span>
              <strong className="font-semibold text-white">
                {totalClasses}
              </strong>{" "}
              Classes
            </span>
            <span className="text-slate-700">&bull;</span>
            <span>
              <strong className="font-semibold text-white">
                {totalSamples}
              </strong>{" "}
              Samples
            </span>
            {hasTrainedModel && (
              <>
                <span className="text-slate-700">&bull;</span>
                <span className="flex items-center gap-1 text-emerald-400">
                  <Activity className="h-3.5 w-3.5" />
                  Model{" "}
                  {trainedAccuracy !== undefined
                    ? `${trainedAccuracy}%`
                    : "Active"}
                </span>
              </>
            )}
          </div>

          <div
            title="All data is saved strictly in your browser's local storage — never uploaded to a server."
            className="flex items-center gap-1.5 rounded-lg border border-slate-800/60 bg-slate-900/50 px-2.5 py-1.5 text-[11px] text-slate-400"
          >
            <HardDrive className="h-3.5 w-3.5 text-emerald-400" />
            <span>
              {syncStatus === "saving"
                ? "Saving to browser…"
                : "Saved in browser"}
            </span>
          </div>
        </div>

        {/* Action Toolbar */}
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenTutorial}
            className="flex items-center gap-1.5 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-300 transition hover:bg-cyan-500/20"
          >
            <BookOpen className="h-3.5 w-3.5" />
            <span>Tutorial</span>
          </button>

          <button
            onClick={onLoadDemoData}
            title="Populate 4 sample gesture classes with 3D landmarks to test training immediately"
            className="flex items-center gap-1.5 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-1.5 text-xs font-medium text-violet-300 transition hover:bg-violet-500/20"
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Demo Data</span>
          </button>

          <button
            onClick={onOpenHub}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/90 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-slate-600 hover:bg-slate-800"
          >
            <FolderArchive className="h-3.5 w-3.5 text-cyan-400" />
            <span className="hidden md:inline">Saved &amp; Export</span>
          </button>

          <button
            onClick={onQuickDownloadZip}
            disabled={totalSamples === 0}
            className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-sm shadow-cyan-500/20 transition hover:from-cyan-400 hover:to-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Download className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Download ZIP</span>
          </button>

          <button
            onClick={onToggleHelp}
            title="Toggle Quick Studio Guide"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-800 bg-slate-900 text-slate-400 transition hover:border-slate-700 hover:text-white"
          >
            <HelpCircle className="h-4 w-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
