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
} from "lucide-react";
import { CapturedSample, PoseClassMeta } from "@/types/handpose";
import { drawHandsOnCanvas } from "@/lib/landmarks";

interface SampleCardProps {
  sample: CapturedSample;
  accentColor: string;
  showOverlay: boolean;
  onDelete: (id: string) => void;
}

function SampleCard({
  sample,
  accentColor,
  showOverlay,
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

  return (
    <div className="group relative aspect-[4/3] overflow-hidden rounded-xl border border-slate-800 bg-slate-950 shadow-sm transition hover:border-slate-600">
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
        <div className="flex h-full w-full items-center justify-center bg-slate-900 text-[10px] text-slate-500">
          3D Landmarks
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={160}
        height={120}
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
      />

      {/* Handedness & Confidence Badge */}
      {primaryHand && (
        <div className="pointer-events-none absolute bottom-1.5 left-1.5 flex items-center gap-1 rounded bg-slate-950/85 px-1.5 py-0.5 text-[10px] font-medium text-slate-200 backdrop-blur-sm">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: accentColor }}
          />
          <span>
            {primaryHand.handedness[0]} &bull;{" "}
            {(primaryHand.score * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Delete Sample Button */}
      <button
        onClick={() => onDelete(sample.id)}
        title="Delete sample"
        className="absolute top-1.5 right-1.5 flex h-6 w-6 items-center justify-center rounded-md bg-rose-600/90 text-white opacity-0 shadow transition group-hover:opacity-100 hover:bg-rose-500"
      >
        <X className="h-3.5 w-3.5" />
      </button>
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
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-4 shadow-xl">
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <FolderPlus className="h-4 w-4 text-cyan-400" />
            <h2 className="text-sm font-bold text-white">
              1. Gesture Classes ({classes.length})
            </h2>
          </div>
          {samples.length > 0 && (
            <button
              onClick={onClearAll}
              className="text-[11px] font-medium text-rose-400 hover:text-rose-300 hover:underline"
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
            placeholder="Enter new gesture class name (e.g. Peace)…"
            className="flex-1 rounded-xl border border-slate-700/80 bg-slate-950 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!newClassInput.trim()}
            className="flex items-center gap-1.5 rounded-xl bg-cyan-500 px-3.5 py-2 text-xs font-bold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Plus className="h-4 w-4" />
            <span>Add Class</span>
          </button>
        </form>

        {/* Quick-Add Preset Chips */}
        {availablePresets.length > 0 && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-slate-500">Quick presets:</span>
            {availablePresets.map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => onAddClass(preset)}
                className="rounded-full border border-slate-800 bg-slate-950/80 px-2.5 py-0.5 text-[11px] font-medium text-slate-300 transition hover:border-cyan-500/40 hover:text-cyan-300"
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
                    ? "border-cyan-500/60 bg-cyan-500/10 shadow-sm shadow-cyan-500/10"
                    : "border-slate-800 bg-slate-950/60 hover:border-slate-700"
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
                        className="w-full rounded border border-cyan-500 bg-slate-900 px-2 py-0.5 text-xs text-white focus:outline-none"
                      />
                      <button
                        onClick={() => commitRename(cls.name)}
                        className="rounded p-1 text-emerald-400 hover:bg-slate-800"
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
                          <span className="rounded bg-cyan-500/20 px-1.5 py-0.2 text-[9px] font-semibold uppercase text-cyan-300">
                            Active
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400">
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
                      className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
                    >
                      <Edit3 className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => onDeleteClass(cls.name)}
                      title="Delete class and its samples"
                      className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-500/20 hover:text-rose-400"
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
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-4 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">
              Captured Samples ({displayedSamples.length})
            </h3>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-[11px]">
              <button
                onClick={() => setGalleryFilter("active")}
                className={`rounded-md px-2 py-0.5 font-medium transition ${
                  galleryFilter === "active"
                    ? "bg-slate-800 text-cyan-300"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                Active Class
              </button>
              <button
                onClick={() => setGalleryFilter("all")}
                className={`rounded-md px-2 py-0.5 font-medium transition ${
                  galleryFilter === "all"
                    ? "bg-slate-800 text-cyan-300"
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
                  ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                  : "border-slate-800 bg-slate-950 text-slate-400"
              }`}
            >
              <Eye className="h-3 w-3" />
              <span>Landmarks</span>
            </button>
          </div>
        </div>

        {displayedSamples.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40 py-8 px-4 text-center">
            <p className="text-xs font-medium text-slate-300">
              No samples captured for{" "}
              {galleryFilter === "active" && activeClassName
                ? `"${activeClassName}"`
                : "this workspace"}{" "}
              yet.
            </p>
            <p className="mt-1 text-[11px] text-slate-500">
              Hold your hand in front of the camera and press{" "}
              <kbd className="rounded border border-slate-700 bg-slate-800 px-1 py-0.5 font-mono text-[10px] text-slate-300">
                Space
              </kbd>{" "}
              or use <strong>Auto-Burst</strong> to record samples quickly.
            </p>
          </div>
        ) : (
          <div className="grid max-h-[290px] grid-cols-3 gap-2.5 overflow-y-auto pr-1 sm:grid-cols-4 md:grid-cols-5">
            {displayedSamples
              .slice()
              .reverse()
              .map((sample) => (
                <SampleCard
                  key={sample.id}
                  sample={sample}
                  accentColor={colorMap.get(sample.className) || "#38bdf8"}
                  showOverlay={showOverlayOnThumbs}
                  onDelete={onDeleteSample}
                />
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
