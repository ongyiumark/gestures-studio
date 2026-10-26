"use client";

import React, { useEffect, useRef, useState } from "react";
import Image from "next/image";
import {
  Check,
  Edit3,
  Eye,
  FolderPlus,
  Layers,
  Plus,
  Trash2,
  X,
  ZoomIn,
} from "lucide-react";
import { CapturedSample, PoseClassMeta } from "@/types/handpose";
import { drawHandsOnCanvas } from "@/lib/landmarks";

interface SampleCardProps {
  sample: CapturedSample;
  accentColor: string;
  showOverlay: boolean;
  onPreview: (sample: CapturedSample) => void;
  onDelete: (id: string) => void;
}

function SampleCard({
  sample,
  accentColor,
  showOverlay,
  onPreview,
  onDelete,
}: SampleCardProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (showOverlay && sample.hands && sample.hands.length > 0) {
      drawHandsOnCanvas(sample.hands, ctx, canvas.width, canvas.height, {
        compact: true,
        showLabels: false,
        accentColor,
      });
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [sample.hands, accentColor, showOverlay]);

  const primaryHand = sample.hands?.[0];
  const handCount = sample.hands?.length || 0;

  return (
    <div
      onClick={() => onPreview(sample)}
      className="group relative aspect-[4/3] overflow-hidden rounded-xl border border-ocean-800/60 bg-[#08121d] shadow-sm transition hover:border-ocean-400 hover:scale-[1.02] cursor-pointer"
      title="Click to blow up & preview sample"
    >
      {sample.imageDataUrl ? (
        <Image
          src={sample.imageDataUrl}
          alt={sample.className}
          fill
          unoptimized
          sizes="160px"
          className="object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-[#0d1a29] text-[10px] text-slate-400">
          3D Landmarks
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={160}
        height={120}
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
      />

      {/* Hover Zoom Hint */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/25 opacity-0 transition group-hover:opacity-100">
        <div className="rounded-full bg-[#0d2238]/90 p-2 text-sky-200 shadow-lg border border-ocean-500/50">
          <ZoomIn className="h-4 w-4" />
        </div>
      </div>

      {/* Handedness & Confidence Badge */}
      {primaryHand && (
        <div className="pointer-events-none absolute bottom-1.5 left-1.5 flex items-center gap-1 rounded bg-[#071320]/90 px-1.5 py-0.5 text-[10px] font-medium text-slate-200 backdrop-blur-sm">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: accentColor }}
          />
          <span>
            {handCount > 1
              ? `2H • ${sample.hands.map((h) => h.handedness[0]).join("+")}`
              : `${primaryHand.handedness[0]} • ${(primaryHand.score * 100).toFixed(0)}%`}
          </span>
        </div>
      )}

      {/* Delete Sample Button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete(sample.id);
        }}
        title="Delete sample"
        className="absolute top-1.5 right-1.5 z-10 flex h-6 w-6 items-center justify-center rounded-md bg-terracotta-700/90 text-white opacity-0 shadow transition group-hover:opacity-100 hover:bg-terracotta-600 active:scale-95"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

interface SamplePreviewModalProps {
  sample: CapturedSample;
  accentColor: string;
  onClose: () => void;
  onDelete: (id: string) => void;
}

function SamplePreviewModal({
  sample,
  accentColor,
  onClose,
  onDelete,
}: SamplePreviewModalProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (sample.hands && sample.hands.length > 0) {
      drawHandsOnCanvas(sample.hands, ctx, canvas.width, canvas.height, {
        compact: false,
        showLabels: true,
        accentColor,
      });
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [sample.hands, accentColor]);

  const dateStr = new Date(sample.timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-3 sm:p-6 backdrop-blur-sm"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative flex flex-col w-full max-w-2xl overflow-hidden rounded-2xl border border-ocean-600/70 bg-[#0d1c2d] shadow-2xl animate-in zoom-in-95 duration-150"
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-ocean-700/60 bg-[#12263a] px-4 py-3">
          <div className="flex items-center gap-2.5">
            <span
              className="h-3.5 w-3.5 rounded-full shadow-sm"
              style={{ backgroundColor: accentColor }}
            />
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-white">{sample.className}</h3>
                <span className="rounded bg-ocean-700/60 px-2 py-0.5 text-[10px] font-mono text-sky-200">
                  {sample.hands.length === 1 ? "1 Hand" : `${sample.hands.length} Hands`}
                </span>
              </div>
              <p className="text-[11px] text-sky-200/70">Captured at {dateStr}</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-[#1a344e] hover:text-white transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Enlarged Image & Canvas Viewport */}
        <div className="relative aspect-[4/3] w-full bg-[#07111b] flex items-center justify-center overflow-hidden">
          {sample.imageDataUrl ? (
            <Image
              src={sample.imageDataUrl}
              alt={sample.className}
              fill
              unoptimized
              className="object-contain"
            />
          ) : (
            <div className="text-xs text-slate-400">No Image Preview Available</div>
          )}

          <canvas
            ref={canvasRef}
            width={sample.width || 640}
            height={sample.height || 480}
            className="pointer-events-none absolute inset-0 h-full w-full object-contain"
          />
        </div>

        {/* Modal Footer */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-ocean-700/60 bg-[#102235] px-4 py-3">
          <div className="flex items-center gap-2 text-xs text-slate-300">
            {sample.hands.map((h, i) => (
              <span
                key={i}
                className="rounded-lg border border-ocean-700/60 bg-[#0a1624] px-2.5 py-1 text-[11px] font-medium"
              >
                {h.handedness}: <strong className="text-emerald-400">{(h.score * 100).toFixed(0)}%</strong>
              </span>
            ))}
            <span className="text-ocean-400 hidden sm:inline">&bull;</span>
            <span className="text-[11px] text-slate-400 hidden sm:inline">
              Resolution: {sample.width}&times;{sample.height}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                onDelete(sample.id);
                onClose();
              }}
              className="flex items-center gap-1.5 rounded-lg border border-rose-500/30 bg-rose-500/15 px-3 py-1.5 text-xs font-semibold text-rose-300 hover:bg-rose-500/25 transition active:scale-95"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Delete Sample</span>
            </button>
            <button
              onClick={onClose}
              className="rounded-lg bg-ocean-700 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-ocean-600 transition active:scale-95"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ClassManagerProps {
  classes: PoseClassMeta[];
  samples: CapturedSample[];
  activeClassName: string | null;
  onSelectClass: (name: string) => void;
  onAddClass: (name: string) => void;
  onRenameClass: (oldName: string, newName: string) => void;
  onDeleteClass: (name: string) => void;
  onDeleteSample: (id: string) => void;
  onClearAll: () => void;
}

const PRESET_CLASSES = [
  "Thumbs Up",
  "Peace",
  "Open Palm",
  "Fist",
  "OK Sign",
  "Pointing",
];

export default function ClassManager({
  classes,
  samples,
  activeClassName,
  onSelectClass,
  onAddClass,
  onRenameClass,
  onDeleteClass,
  onDeleteSample,
  onClearAll,
}: ClassManagerProps) {
  const [newClassInput, setNewClassInput] = useState("");
  const [editingClass, setEditingClass] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [showOverlayOnThumbs, setShowOverlayOnThumbs] = useState(true);
  const [galleryFilter, setGalleryFilter] = useState<"active" | "all">("active");
  const [previewSample, setPreviewSample] = useState<CapturedSample | null>(null);

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newClassInput.trim();
    if (!trimmed) return;
    onAddClass(trimmed);
    setNewClassInput("");
  };

  const startRename = (name: string) => {
    setEditingClass(name);
    setEditValue(name);
  };

  const commitRename = (oldName: string) => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== oldName) {
      onRenameClass(oldName, trimmed);
    }
    setEditingClass(null);
  };

  const countsByClass = new Map<string, number>();
  for (const s of samples) {
    countsByClass.set(s.className, (countsByClass.get(s.className) || 0) + 1);
  }

  const displayedSamples =
    galleryFilter === "all" || !activeClassName
      ? samples
      : samples.filter((s) => s.className === activeClassName);

  const colorMap = new Map(classes.map((c) => [c.name, c.color]));
  const existingNames = new Set(classes.map((c) => c.name.toLowerCase()));
  const availablePresets = PRESET_CLASSES.filter(
    (p) => !existingNames.has(p.toLowerCase())
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Pose Classes Card */}
      <div className="rounded-2xl border border-ocean-700/50 bg-[#101e2e]/90 p-4 shadow-xl">
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <FolderPlus className="h-4 w-4 text-terracotta-400" />
            <h2 className="text-sm font-bold text-white">
              1. Hand Pose Classes ({classes.length})
            </h2>
          </div>
          {samples.length > 0 && (
            <button
              onClick={onClearAll}
              className="text-[11px] font-medium text-terracotta-400 hover:text-terracotta-300 hover:underline"
            >
              Reset Workspace
            </button>
          )}
        </div>

        {/* Add Class Input */}
        <form onSubmit={handleCreate} className="flex gap-2">
          <input
            type="text"
            value={newClassInput}
            onChange={(e) => setNewClassInput(e.target.value)}
            placeholder="Enter new hand pose class name (e.g. Peace)…"
            className="flex-1 rounded-xl border border-ocean-700/60 bg-[#081320] px-3 py-2 text-xs text-white placeholder-slate-400 focus:border-ocean-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!newClassInput.trim()}
            className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-terracotta-500 to-terracotta-600 px-3.5 py-2 text-xs font-bold text-white shadow-sm shadow-terracotta-950/40 transition hover:from-terracotta-400 hover:to-terracotta-500 disabled:cursor-not-allowed disabled:opacity-40 active:scale-95"
          >
            <Plus className="h-4 w-4" />
            <span>Add Class</span>
          </button>
        </form>

        {/* Quick-Add Preset Chips */}
        {availablePresets.length > 0 && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-slate-400">Quick presets:</span>
            {availablePresets.map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => onAddClass(preset)}
                className="rounded-full border border-ocean-700/60 bg-[#091522] px-2.5 py-0.5 text-[11px] font-medium text-sky-200 transition hover:border-ocean-500 hover:text-white"
              >
                + {preset}
              </button>
            ))}
          </div>
        )}

        {/* Class List */}
        <div className="mt-3.5 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {classes.map((cls) => {
            const count = countsByClass.get(cls.name) || 0;
            const isSelected = cls.name === activeClassName;
            const isEditing = editingClass === cls.name;

            return (
              <div
                key={cls.name}
                onClick={() => !isEditing && onSelectClass(cls.name)}
                className={`group flex cursor-pointer items-center justify-between gap-2 rounded-xl border p-2.5 transition ${
                  isSelected
                    ? "border-terracotta-500/70 bg-[#162e47] shadow-sm shadow-terracotta-950/20"
                    : "border-ocean-800/60 bg-[#0a1624]/80 hover:border-ocean-700"
                }`}
              >
                <div className="flex min-w-0 flex-1 items-center gap-2.5">
                  <span
                    className="h-3 w-3 shrink-0 rounded-full shadow-sm"
                    style={{ backgroundColor: cls.color }}
                  />
                  {isEditing ? (
                    <div
                      className="flex flex-1 items-center gap-1"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitRename(cls.name);
                          if (e.key === "Escape") setEditingClass(null);
                        }}
                        autoFocus
                        className="w-full rounded border border-ocean-500 bg-[#091522] px-2 py-0.5 text-xs text-white focus:outline-none"
                      />
                      <button
                        onClick={() => commitRename(cls.name)}
                        className="rounded p-1 text-emerald-400 hover:bg-[#16304d]"
                      >
                        <Check className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ) : (
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate text-xs font-bold text-white">
                          {cls.name}
                        </span>
                        {isSelected && (
                          <span className="rounded bg-terracotta-500/20 px-1.5 py-0.2 text-[9px] font-bold uppercase tracking-wider text-terracotta-300">
                            Active
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-sky-200/70">
                        {count} {count === 1 ? "sample" : "samples"}
                      </div>
                    </div>
                  )}
                </div>

                {/* Class Actions */}
                {!isEditing && (
                  <div
                    className="flex items-center gap-1 opacity-70 group-hover:opacity-100"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      onClick={() => startRename(cls.name)}
                      title="Rename class"
                      className="rounded-lg p-1.5 text-slate-400 hover:bg-[#16304d] hover:text-white"
                    >
                      <Edit3 className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => onDeleteClass(cls.name)}
                      title="Delete class and its samples"
                      className="rounded-lg p-1.5 text-slate-400 hover:bg-terracotta-500/20 hover:text-terracotta-300"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Captured Samples Gallery Card */}
      <div className="rounded-2xl border border-ocean-700/50 bg-[#101e2e]/90 p-4 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-sky-300" />
            <h3 className="text-sm font-bold text-white">
              Captured Samples ({displayedSamples.length})
            </h3>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-ocean-700/60 bg-[#08121d] p-0.5 text-[11px]">
              <button
                onClick={() => setGalleryFilter("active")}
                className={`rounded-md px-2.5 py-0.5 font-medium transition ${
                  galleryFilter === "active"
                    ? "bg-ocean-700 text-white font-semibold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                Active Class
              </button>
              <button
                onClick={() => setGalleryFilter("all")}
                className={`rounded-md px-2.5 py-0.5 font-medium transition ${
                  galleryFilter === "all"
                    ? "bg-ocean-700 text-white font-semibold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                All ({samples.length})
              </button>
            </div>

            <button
              onClick={() => setShowOverlayOnThumbs((v) => !v)}
              title="Toggle 21-keypoint skeleton on sample thumbnails"
              className={`flex items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-medium transition ${
                showOverlayOnThumbs
                  ? "border-ocean-500/50 bg-ocean-600/20 text-sky-200"
                  : "border-ocean-800/60 bg-[#08121d] text-slate-400"
              }`}
            >
              <Eye className="h-3 w-3" />
              <span>Landmarks</span>
            </button>
          </div>
        </div>

        {displayedSamples.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-ocean-800/80 bg-[#081320]/60 py-8 px-4 text-center">
            <p className="text-xs font-medium text-slate-200">
              No samples captured for{" "}
              {galleryFilter === "active" && activeClassName
                ? `"${activeClassName}"`
                : "this workspace"}{" "}
              yet.
            </p>
            <p className="mt-1 text-[11px] text-sky-200/70">
              Hold your hand in front of the camera and press{" "}
              <kbd className="rounded border border-ocean-600/50 bg-[#091522] px-1 py-0.5 font-mono text-[10px] text-slate-200">
                Space
              </kbd>{" "}
              or use <strong>Auto-Burst</strong> to record samples quickly.
            </p>
          </div>
        ) : (
          <div className="grid max-h-[300px] grid-cols-2 min-[420px]:grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2.5 overflow-y-auto pr-1">
            {displayedSamples
              .slice()
              .reverse()
              .map((sample) => (
                <SampleCard
                  key={sample.id}
                  sample={sample}
                  accentColor={colorMap.get(sample.className) || "#38bdf8"}
                  showOverlay={showOverlayOnThumbs}
                  onPreview={setPreviewSample}
                  onDelete={onDeleteSample}
                />
              ))}
          </div>
        )}
      </div>

      {/* Enlarged Sample Preview Modal */}
      {previewSample && (
        <SamplePreviewModal
          sample={previewSample}
          accentColor={colorMap.get(previewSample.className) || "#38bdf8"}
          onClose={() => setPreviewSample(null)}
          onDelete={onDeleteSample}
        />
      )}
    </div>
  );
}
