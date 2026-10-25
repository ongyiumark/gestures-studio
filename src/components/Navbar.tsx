"use client";

import React from "react";
import Image from "next/image";
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
    <header className="sticky top-0 z-30 border-b border-ocean-700/50 bg-gradient-to-r from-[#173857]/95 via-[#286398]/90 to-[#193d60]/95 backdrop-blur-md shadow-md shadow-black/20">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-3.5 py-2.5 sm:px-6 sm:py-3">
        {/* Brand Identity */}
        <div className="flex items-center gap-2.5 sm:gap-3">
          <div className="relative h-9 w-9 sm:h-10 sm:w-10 shrink-0 overflow-hidden rounded-xl border border-ocean-600/40 shadow-md shadow-terracotta-950/40">
            <Image
              src="/icon.png"
              alt="Hand Pose Studio Icon"
              width={40}
              height={40}
              className="h-full w-full object-cover"
              priority
            />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-white drop-shadow-[1px_1px_3px_#FF8A55] sm:text-lg">
                Hand Pose Studio
              </h1>
              <span className="rounded-full border border-gold-300/40 bg-gold-200/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-gold-200 shadow-sm">
                v2.0
              </span>
            </div>
            <p className="text-[11px] text-sky-100/85 sm:text-xs">
              3D Hand Pose Dataset &amp; In-Browser Classifier &bull; by{" "}
              <a
                href="https://github.com/ongyiumark/"
                target="_blank"
                rel="noreferrer"
                className="font-medium text-[#FFE79C] hover:text-[#EBBE30] transition underline underline-offset-2"
              >
                ongyiumark
              </a>
            </p>
          </div>
        </div>

        {/* Live Telemetry Badges (Desktop & Tablet) */}
        <div className="hidden items-center gap-2 lg:flex">
          <div className="flex items-center gap-2 rounded-lg border border-ocean-700/60 bg-[#0e2135]/80 px-3 py-1.5 text-xs backdrop-blur-sm">
            {modelReady ? (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
                <span className="font-medium text-slate-100">
                  Tracker Ready
                </span>
                <span className="text-ocean-400">|</span>
                <span className="font-mono text-gold-200">{fps} FPS</span>
              </>
            ) : (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-300" />
                <span className="text-amber-200 font-medium">Loading 3D Tracker…</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2.5 rounded-lg border border-ocean-700/60 bg-[#0e2135]/80 px-3 py-1.5 text-xs text-slate-200 backdrop-blur-sm">
            <span>
              <strong className="font-semibold text-white">
                {totalClasses}
              </strong>{" "}
              Classes
            </span>
            <span className="text-ocean-400">&bull;</span>
            <span>
              <strong className="font-semibold text-white">
                {totalSamples}
              </strong>{" "}
              Samples
            </span>
            {hasTrainedModel && (
              <>
                <span className="text-ocean-400">&bull;</span>
                <span className="flex items-center gap-1 font-medium text-emerald-300">
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
            className="flex items-center gap-1.5 rounded-lg border border-ocean-700/50 bg-[#0e2135]/60 px-2.5 py-1.5 text-[11px] text-sky-200/80"
          >
            <HardDrive className="h-3.5 w-3.5 text-emerald-400" />
            <span>
              {syncStatus === "saving"
                ? "Saving to browser…"
                : "Saved locally"}
            </span>
          </div>
        </div>

        {/* Action Toolbar (Responsive across Mobile, Tablet, Desktop) */}
        <div className="flex items-center gap-1.5 sm:gap-2">
          {/* Mobile status indicator badge if on small screens */}
          <div className="flex lg:hidden items-center gap-1.5 rounded-lg border border-ocean-700/50 bg-[#0e2135]/80 px-2 py-1 text-[11px]">
            <span
              className={`h-2 w-2 rounded-full ${
                modelReady
                  ? "bg-emerald-400 shadow-[0_0_6px_#34d399]"
                  : "bg-amber-400 animate-pulse"
              }`}
            />
            <span className="font-mono text-slate-200">{fps} FPS</span>
          </div>

          <button
            onClick={onOpenTutorial}
            title="Open guided walkthrough tutorial"
            className="flex items-center gap-1.5 rounded-lg border border-gold-300/40 bg-gold-200/15 px-2.5 py-1.5 text-xs font-semibold text-gold-100 transition hover:bg-gold-200/25 active:scale-95"
          >
            <BookOpen className="h-3.5 w-3.5" />
            <span className="hidden min-[380px]:inline">Tutorial</span>
          </button>

          <button
            onClick={onLoadDemoData}
            title="Populate 4 sample hand pose classes with 3D landmarks to test training immediately"
            className="flex items-center gap-1.5 rounded-lg border border-white/15 bg-white/10 px-2.5 py-1.5 text-xs font-medium text-slate-100 transition hover:bg-white/20 active:scale-95"
          >
            <Sparkles className="h-3.5 w-3.5 text-gold-200" />
            <span className="hidden sm:inline">Demo Data</span>
          </button>

          <button
            onClick={onOpenHub}
            title="View saved browser datasets, models, and export options"
            className="flex items-center gap-1.5 rounded-lg border border-ocean-600/70 bg-[#163554]/90 px-2.5 py-1.5 text-xs font-medium text-slate-100 transition hover:bg-[#1f456c] active:scale-95"
          >
            <FolderArchive className="h-3.5 w-3.5 text-sky-300" />
            <span className="hidden md:inline">Saved &amp; Export</span>
          </button>

          <button
            onClick={onQuickDownloadZip}
            disabled={totalSamples === 0}
            title="Download complete dataset as a ZIP (images + 3D landmarks + CSV)"
            className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-terracotta-500 to-terracotta-600 px-3 py-1.5 text-xs font-bold text-white shadow-sm shadow-terracotta-950/40 transition hover:from-terracotta-400 hover:to-terracotta-500 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Download className="h-3.5 w-3.5" />
            <span className="hidden min-[480px]:inline">Download ZIP</span>
          </button>

          <button
            onClick={onToggleHelp}
            title="Toggle Quick Studio Guide"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-ocean-700/60 bg-[#0e2135]/80 text-sky-200 transition hover:bg-[#163554] hover:text-white"
          >
            <HelpCircle className="h-4 w-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
