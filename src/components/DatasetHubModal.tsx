"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  Archive,
  BrainCircuit,
  Download,
  FileJson,
  FileSpreadsheet,
  FolderOpen,
  HardDrive,
  Loader2,
  Save,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  CapturedSample,
  DatasetSnapshotSummary,
  PoseClassMeta,
  TrainedModelRecord,
} from "@/types/handpose";
import {
  deleteDatasetLocally,
  deleteModelLocally,
  listLocalDatasets,
  listLocalModels,
  loadDatasetLocallyById,
  saveDatasetLocally,
} from "@/lib/storage";
import {
  exportDatasetJson,
  exportDatasetZip,
  exportLandmarksCsv,
  exportTrainedModelZip,
  parseImportedDatasetJson,
} from "@/lib/export";

interface DatasetHubModalProps {
  isOpen: boolean;
  onClose: () => void;
  classes: PoseClassMeta[];
  samples: CapturedSample[];
  activeModel: TrainedModelRecord | null;
  onRestoreWorkspace: (
    classes: PoseClassMeta[],
    samples: CapturedSample[]
  ) => void;
  onRestoreModel: (model: TrainedModelRecord) => void;
}

export default function DatasetHubModal({
  isOpen,
  onClose,
  classes,
  samples,
  activeModel,
  onRestoreWorkspace,
  onRestoreModel,
}: DatasetHubModalProps) {
  const [activeTab, setActiveTab] = useState<"export" | "local">("export");
  const [datasetName, setDatasetName] = useState("gesture-dataset");
  const [snapshotDesc, setSnapshotDesc] = useState("");

  const [savedDatasets, setSavedDatasets] = useState<DatasetSnapshotSummary[]>(
    []
  );
  const [savedModels, setSavedModels] = useState<TrainedModelRecord[]>([]);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [savingSnapshot, setSavingSnapshot] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refreshLocalData = async () => {
    try {
      setLoadingLocal(true);
      const [dsList, mdlList] = await Promise.all([
        listLocalDatasets(),
        listLocalModels(),
      ]);
      setSavedDatasets(dsList);
      setSavedModels(mdlList);
    } catch (err) {
      console.warn("Could not load local browser records:", err);
    } finally {
      setLoadingLocal(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      refreshLocalData();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const showTempMessage = (msg: string) => {
    setStatusMsg(msg);
    setTimeout(() => setStatusMsg(null), 3000);
  };

  const handleSaveSnapshot = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!datasetName.trim() || samples.length === 0) return;

    try {
      setSavingSnapshot(true);
      await saveDatasetLocally({
        name: datasetName.trim(),
        description: snapshotDesc.trim(),
        classes,
        samples,
      });
      setSnapshotDesc("");
      await refreshLocalData();
      showTempMessage("Dataset snapshot saved to your browser's local storage!");
    } catch (err) {
      showTempMessage(
        err instanceof Error ? err.message : "Failed to save local snapshot"
      );
    } finally {
      setSavingSnapshot(false);
    }
  };

  const handleLoadDataset = async (id: string) => {
    try {
      setLoadingLocal(true);
      const snap = await loadDatasetLocallyById(id);
      onRestoreWorkspace(snap.classes, snap.samples);
      showTempMessage(
        `Loaded dataset "${snap.name}" (${snap.samples.length} samples)`
      );
      onClose();
    } catch {
      showTempMessage("Failed to load dataset from local storage.");
    } finally {
      setLoadingLocal(false);
    }
  };

  const handleDeleteDataset = async (id: string) => {
    try {
      await deleteDatasetLocally(id);
      setSavedDatasets((prev) => prev.filter((d) => d.id !== id));
    } catch {
      showTempMessage("Failed to delete local dataset.");
    }
  };

  const handleDeleteModel = async (id: string) => {
    try {
      await deleteModelLocally(id);
      setSavedModels((prev) => prev.filter((m) => m.id !== id));
    } catch {
      showTempMessage("Failed to delete local model.");
    }
  };

  const handleImportJsonFile = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const imported = parseImportedDatasetJson(text);
      onRestoreWorkspace(imported.classes, imported.samples);
      showTempMessage(
        `Imported ${imported.samples.length} samples across ${imported.classes.length} classes!`
      );
      onClose();
    } catch (err) {
      showTempMessage(
        err instanceof Error ? err.message : "Invalid dataset JSON file"
      );
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm">
      <div className="flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
          <div className="flex items-center gap-2.5">
            <HardDrive className="h-5 w-5 text-cyan-400" />
            <div>
              <h2 className="text-base font-bold text-white">
                Export Files &amp; Browser Local Snapshots
              </h2>
              <p className="text-xs text-slate-400">
                Download datasets/models or manage snapshots stored privately in
                your browser
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Modal Tabs */}
        <div className="flex border-b border-slate-800 bg-slate-950/60 px-5 pt-2">
          <button
            onClick={() => setActiveTab("export")}
            className={`border-b-2 px-4 py-2 text-xs font-bold transition ${
              activeTab === "export"
                ? "border-cyan-400 text-cyan-300"
                : "border-transparent text-slate-400 hover:text-white"
            }`}
          >
            Export &amp; Import Files
          </button>
          <button
            onClick={() => setActiveTab("local")}
            className={`border-b-2 px-4 py-2 text-xs font-bold transition ${
              activeTab === "local"
                ? "border-cyan-400 text-cyan-300"
                : "border-transparent text-slate-400 hover:text-white"
            }`}
          >
            Saved in Browser ({savedDatasets.length + savedModels.length})
          </button>
        </div>

        {statusMsg && (
          <div className="bg-cyan-500/15 border-b border-cyan-500/30 px-5 py-2 text-xs font-medium text-cyan-300">
            {statusMsg}
          </div>
        )}

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {activeTab === "export" ? (
            <div className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Export Filename Prefix
                </label>
                <input
                  type="text"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {/* 1. Full ZIP Archive */}
                <button
                  onClick={() => exportDatasetZip(classes, samples, datasetName)}
                  disabled={samples.length === 0}
                  className="flex flex-col items-start gap-1.5 rounded-xl border border-slate-800 bg-slate-950/80 p-4 text-left transition hover:border-cyan-500/50 hover:bg-slate-950 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <div className="flex items-center gap-2 text-cyan-400 font-bold text-xs">
                    <Archive className="h-4 w-4" />
                    <span>Download Full Dataset (.zip)</span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Includes class folders with JPEG images, JSON 21-point
                    annotations, and landmarks.csv.
                  </p>
                </button>

                {/* 2. Landmarks CSV */}
                <button
                  onClick={() => exportLandmarksCsv(samples, datasetName)}
                  disabled={samples.length === 0}
                  className="flex flex-col items-start gap-1.5 rounded-xl border border-slate-800 bg-slate-950/80 p-4 text-left transition hover:border-emerald-500/50 hover:bg-slate-950 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs">
                    <FileSpreadsheet className="h-4 w-4" />
                    <span>Export Landmarks Table (.csv)</span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Flat tabular CSV with raw &amp; normalized 63-D coordinates
                    for Python / scikit-learn / PyTorch.
                  </p>
                </button>

                {/* 3. Workspace JSON Backup */}
                <button
                  onClick={() => exportDatasetJson(classes, samples, datasetName)}
                  disabled={samples.length === 0}
                  className="flex flex-col items-start gap-1.5 rounded-xl border border-slate-800 bg-slate-950/80 p-4 text-left transition hover:border-violet-500/50 hover:bg-slate-950 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <div className="flex items-center gap-2 text-violet-400 font-bold text-xs">
                    <FileJson className="h-4 w-4" />
                    <span>Backup Workspace (.json)</span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Lossless JSON snapshot of all classes, images, and 3D
                    landmarks for later restoration.
                  </p>
                </button>

                {/* 4. Import Workspace JSON */}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex flex-col items-start gap-1.5 rounded-xl border border-dashed border-slate-700 bg-slate-950/50 p-4 text-left transition hover:border-amber-400/60 hover:bg-slate-950"
                >
                  <div className="flex items-center gap-2 text-amber-300 font-bold text-xs">
                    <Upload className="h-4 w-4" />
                    <span>Import Workspace (.json)</span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Restore a previously exported HandPose Trainer JSON dataset
                    into the studio.
                  </p>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,application/json"
                  onChange={handleImportJsonFile}
                  className="hidden"
                />
              </div>

              {/* Active Trained Model Export */}
              {activeModel?.serializedModel && (
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <h4 className="text-xs font-bold text-emerald-300">
                        Trained TensorFlow.js Model Ready (
                        {activeModel.finalAccuracy}% Acc)
                      </h4>
                      <p className="text-[11px] text-slate-400">
                        Bundles model.json, model.weights.bin, classes.json &amp;
                        inference-example.js
                      </p>
                    </div>
                    <button
                      onClick={() => exportTrainedModelZip(activeModel)}
                      className="rounded-lg bg-emerald-500 px-3.5 py-2 text-xs font-bold text-slate-950 hover:bg-emerald-400"
                    >
                      Download TF.js Bundle (.zip)
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-5">
              {/* Save Snapshot to Browser Local Storage */}
              <form
                onSubmit={handleSaveSnapshot}
                className="rounded-xl border border-slate-800 bg-slate-950/80 p-4"
              >
                <h3 className="text-xs font-bold text-white mb-2.5 flex items-center gap-1.5">
                  <Save className="h-4 w-4 text-cyan-400" />
                  <span>
                    Save Named Snapshot in Browser ({samples.length} samples)
                  </span>
                </h3>
                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  <input
                    type="text"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    placeholder="Dataset snapshot name…"
                    className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-white"
                  />
                  <input
                    type="text"
                    value={snapshotDesc}
                    onChange={(e) => setSnapshotDesc(e.target.value)}
                    placeholder="Optional notes / description…"
                    className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-white"
                  />
                </div>
                <button
                  type="submit"
                  disabled={savingSnapshot || samples.length === 0}
                  className="mt-2.5 flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3.5 py-1.5 text-xs font-bold text-slate-950 hover:bg-cyan-400 disabled:opacity-40"
                >
                  {savingSnapshot ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Save className="h-3.5 w-3.5" />
                  )}
                  <span>Save Snapshot Locally</span>
                </button>
              </form>

              {/* Saved Local Datasets */}
              <div>
                <h4 className="text-xs font-bold text-slate-200 mb-2 flex items-center gap-1.5">
                  <FolderOpen className="h-4 w-4 text-cyan-400" />
                  <span>
                    Saved Browser Dataset Snapshots ({savedDatasets.length})
                  </span>
                </h4>

                {loadingLocal ? (
                  <div className="flex items-center gap-2 py-4 text-xs text-slate-400">
                    <Loader2 className="h-4 w-4 animate-spin text-cyan-400" />
                    <span>Reading browser storage…</span>
                  </div>
                ) : savedDatasets.length === 0 ? (
                  <p className="rounded-xl border border-dashed border-slate-800 p-4 text-center text-xs text-slate-500">
                    No named dataset snapshots saved in your browser yet. (Your
                    active workspace is still auto-saved automatically!)
                  </p>
                ) : (
                  <div className="flex flex-col gap-2">
                    {savedDatasets.map((ds) => (
                      <div
                        key={ds.id}
                        className="flex items-center justify-between gap-2 rounded-xl border border-slate-800 bg-slate-950/70 p-3"
                      >
                        <div>
                          <div className="text-xs font-bold text-white">
                            {ds.name}
                          </div>
                          <div className="text-[11px] text-slate-400" suppressHydrationWarning>
                            {ds.sampleCount} samples &bull; {ds.classes.length}{" "}
                            classes &bull;{" "}
                            {new Date(ds.updatedAt).toLocaleDateString("en-US", {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </div>
                          {ds.description && (
                            <div className="text-[11px] text-slate-500">
                              {ds.description}
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => handleLoadDataset(ds.id)}
                            className="flex items-center gap-1 rounded-lg bg-cyan-500/20 px-2.5 py-1 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/30"
                          >
                            <Download className="h-3.5 w-3.5" />
                            <span>Load</span>
                          </button>
                          <button
                            onClick={() => handleDeleteDataset(ds.id)}
                            className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-500/20 hover:text-rose-400"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Saved Local Models */}
              <div>
                <h4 className="text-xs font-bold text-slate-200 mb-2 flex items-center gap-1.5">
                  <BrainCircuit className="h-4 w-4 text-violet-400" />
                  <span>Saved Browser Models ({savedModels.length})</span>
                </h4>

                {savedModels.length === 0 ? (
                  <p className="rounded-xl border border-dashed border-slate-800 p-4 text-center text-xs text-slate-500">
                    No trained models saved to your browser library yet.
                  </p>
                ) : (
                  <div className="flex flex-col gap-2">
                    {savedModels.map((mdl) => (
                      <div
                        key={mdl.id}
                        className="flex items-center justify-between gap-2 rounded-xl border border-slate-800 bg-slate-950/70 p-3"
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-white">
                              {mdl.name}
                            </span>
                            <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                              {mdl.finalAccuracy}% Acc
                            </span>
                          </div>
                          <div className="text-[11px] text-slate-400" suppressHydrationWarning>
                            Classes: {mdl.classes.join(", ")} &bull;{" "}
                            {new Date(mdl.trainedAt).toLocaleDateString("en-US", {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => {
                              onRestoreModel(mdl);
                              onClose();
                            }}
                            className="flex items-center gap-1 rounded-lg bg-violet-500/20 px-2.5 py-1 text-xs font-semibold text-violet-300 hover:bg-violet-500/30"
                          >
                            <Download className="h-3.5 w-3.5" />
                            <span>Activate</span>
                          </button>
                          <button
                            onClick={() => handleDeleteModel(mdl.id)}
                            className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-500/20 hover:text-rose-400"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
