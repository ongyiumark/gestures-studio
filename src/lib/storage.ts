import {
  CapturedSample,
  DatasetSnapshot,
  DatasetSnapshotSummary,
  PoseClassMeta,
  TrainedModelRecord,
} from "@/types/handpose";

const DB_NAME = "handpose_studio_local_v2";
const LEGACY_DB_NAME = "gesture_studio_local_v2";
const DB_VERSION = 2;
const STORE_WORKSPACE = "workspace";
const STORE_DATASETS = "saved_datasets";
const STORE_MODELS = "saved_models";

const LOCAL_STORAGE_PREFS_KEY = "handpose_studio_prefs_v2";
const LEGACY_STORAGE_PREFS_KEY = "gesture_studio_prefs_v2";
const LOCAL_STORAGE_SUMMARY_KEY = "handpose_studio_workspace_summary_v2";

export interface UserPreferences {
  hasCompletedTutorial: boolean;
  hideQuickGuide: boolean;
}

export interface PersistedWorkspace {
  id: "active";
  updatedAt: number;
  classes: PoseClassMeta[];
  samples: CapturedSample[];
  activeModel: TrainedModelRecord | null;
}

/**
 * Reads lightweight user UI preferences from browser localStorage.
 */
export function loadPreferencesFromLocalStorage(): UserPreferences {
  if (typeof window === "undefined") {
    return { hasCompletedTutorial: false, hideQuickGuide: false };
  }
  try {
    const raw =
      window.localStorage.getItem(LOCAL_STORAGE_PREFS_KEY) ||
      window.localStorage.getItem(LEGACY_STORAGE_PREFS_KEY);
    if (!raw) return { hasCompletedTutorial: false, hideQuickGuide: false };
    const parsed = JSON.parse(raw);
    return {
      hasCompletedTutorial: Boolean(parsed.hasCompletedTutorial),
      hideQuickGuide: Boolean(parsed.hideQuickGuide),
    };
  } catch {
    return { hasCompletedTutorial: false, hideQuickGuide: false };
  }
}

/**
 * Saves lightweight user UI preferences to browser localStorage.
 */
export function savePreferencesToLocalStorage(
  prefs: Partial<UserPreferences>
): void {
  if (typeof window === "undefined") return;
  try {
    const current = loadPreferencesFromLocalStorage();
    const updated = { ...current, ...prefs };
    window.localStorage.setItem(
      LOCAL_STORAGE_PREFS_KEY,
      JSON.stringify(updated)
    );
  } catch (err) {
    console.warn("Failed to write preferences to localStorage:", err);
  }
}

function openDatabase(dbName: string = DB_NAME): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined" || !window.indexedDB) {
      reject(new Error("Browser storage is not available in this environment."));
      return;
    }

    const request = window.indexedDB.open(dbName, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_WORKSPACE)) {
        db.createObjectStore(STORE_WORKSPACE, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(STORE_DATASETS)) {
        db.createObjectStore(STORE_DATASETS, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(STORE_MODELS)) {
        db.createObjectStore(STORE_MODELS, { keyPath: "id" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

let cachedDbPromise: Promise<IDBDatabase> | null = null;

function getDatabase(): Promise<IDBDatabase> {
  if (!cachedDbPromise) {
    cachedDbPromise = openDatabase(DB_NAME)
      .then((db) => {
        db.onversionchange = () => {
          db.close();
          cachedDbPromise = null;
        };
        db.onclose = () => {
          cachedDbPromise = null;
        };
        return db;
      })
      .catch((err) => {
        cachedDbPromise = null;
        throw err;
      });
  }
  return cachedDbPromise;
}

/**
 * Saves the active workspace (classes, captured samples, active trained model)
 * 100% locally in the user's browser (localStorage summary + IndexedDB for images/weights).
 */
export async function saveWorkspaceLocally(
  classes: PoseClassMeta[],
  samples: CapturedSample[],
  activeModel: TrainedModelRecord | null
): Promise<void> {
  if (typeof window === "undefined") return;

  // 1. Save lightweight metadata summary to localStorage
  try {
    window.localStorage.setItem(
      LOCAL_STORAGE_SUMMARY_KEY,
      JSON.stringify({
        updatedAt: Date.now(),
        classes,
        sampleCount: samples.length,
        activeModelName: activeModel?.name || null,
        activeModelAccuracy: activeModel?.finalAccuracy || null,
      })
    );
  } catch {
    // Ignore quota warnings on localStorage summary
  }

  // 2. Save full samples (with base64 thumbnails + 3D landmarks) & model weights to browser IndexedDB
  try {
    const db = await getDatabase();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_WORKSPACE, "readwrite");
      const store = tx.objectStore(STORE_WORKSPACE);
      const record: PersistedWorkspace = {
        id: "active",
        updatedAt: Date.now(),
        classes,
        samples,
        activeModel,
      };
      const req = store.put(record);
      req.onerror = () => reject(req.error);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error || new Error("Transaction aborted"));
    });
  } catch (err) {
    console.error("Local browser storage save failed:", err);
    throw err;
  }
}

/**
 * Loads the persisted workspace from the user's browser on startup.
 */
export async function loadWorkspaceLocally(): Promise<PersistedWorkspace | null> {
  try {
    const db = await getDatabase();
    const active = await new Promise<PersistedWorkspace | null>((resolve, reject) => {
      const tx = db.transaction(STORE_WORKSPACE, "readonly");
      const store = tx.objectStore(STORE_WORKSPACE);
      const req = store.get("active");
      req.onsuccess = () => {
        resolve((req.result as PersistedWorkspace) || null);
      };
      req.onerror = () => {
        reject(req.error);
      };
    });

    if (active && (active.classes?.length > 0 || active.samples?.length > 0)) {
      return active;
    }

    // Fallback: check legacy gesture_studio IndexedDB if user previously had data saved
    try {
      if (typeof window !== "undefined" && window.indexedDB) {
        if (typeof window.indexedDB.databases === "function") {
          const existingDbs = await window.indexedDB.databases();
          const hasLegacy = existingDbs.some((db) => db.name === LEGACY_DB_NAME);
          if (!hasLegacy) {
            return active;
          }
        }
        const legacyDb = await openDatabase(LEGACY_DB_NAME);
        const legacyActive = await new Promise<PersistedWorkspace | null>((resolve, reject) => {
          const tx = legacyDb.transaction(STORE_WORKSPACE, "readonly");
          const store = tx.objectStore(STORE_WORKSPACE);
          const req = store.get("active");
          req.onsuccess = () => {
            legacyDb.close();
            resolve((req.result as PersistedWorkspace) || null);
          };
          req.onerror = () => {
            legacyDb.close();
            reject(req.error);
          };
        });

        if (legacyActive && (legacyActive.classes?.length > 0 || legacyActive.samples?.length > 0)) {
          await saveWorkspaceLocally(
            legacyActive.classes,
            legacyActive.samples,
            legacyActive.activeModel
          );
          return legacyActive;
        }
      }
    } catch {
      // Legacy database not present
    }

    return active;
  } catch (err) {
    console.warn("Local browser storage load skipped:", err);
    return null;
  }
}

/**
 * Lists all saved dataset snapshots stored locally in the user's browser.
 */
export async function listLocalDatasets(): Promise<DatasetSnapshotSummary[]> {
  try {
    const db = await getDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_DATASETS, "readonly");
      const store = tx.objectStore(STORE_DATASETS);
      const req = store.getAll();
      req.onsuccess = () => {
        const all = (req.result as DatasetSnapshot[]) || [];
        const summaries: DatasetSnapshotSummary[] = all
          .map((snap) => ({
            id: snap.id,
            name: snap.name,
            description: snap.description,
            createdAt: snap.createdAt,
            updatedAt: snap.updatedAt,
            sampleCount: Array.isArray(snap.samples) ? snap.samples.length : 0,
            classes: snap.classes || [],
          }))
          .sort((a, b) => b.updatedAt - a.updatedAt);
        resolve(summaries);
      };
      req.onerror = () => {
        reject(req.error);
      };
    });
  } catch {
    return [];
  }
}

/**
 * Saves a named dataset snapshot locally in the user's browser.
 */
export async function saveDatasetLocally(params: {
  name: string;
  description: string;
  classes: PoseClassMeta[];
  samples: CapturedSample[];
}): Promise<DatasetSnapshotSummary> {
  const db = await getDatabase();
  const now = Date.now();
  const snapshot: DatasetSnapshot = {
    id: `ds_${now}_${Math.random().toString(36).slice(2, 8)}`,
    name: params.name.trim(),
    description: params.description.trim(),
    createdAt: now,
    updatedAt: now,
    classes: params.classes,
    samples: params.samples,
  };

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DATASETS, "readwrite");
    const store = tx.objectStore(STORE_DATASETS);
    store.put(snapshot);
    tx.oncomplete = () => {
      resolve({
        id: snapshot.id,
        name: snapshot.name,
        description: snapshot.description,
        createdAt: snapshot.createdAt,
        updatedAt: snapshot.updatedAt,
        sampleCount: snapshot.samples.length,
        classes: snapshot.classes,
      });
    };
    tx.onerror = () => {
      reject(tx.error);
    };
    tx.onabort = () => {
      reject(tx.error || new Error("Transaction aborted"));
    };
  });
}

/**
 * Loads a saved dataset snapshot by ID from the user's browser storage.
 */
export async function loadDatasetLocallyById(
  id: string
): Promise<DatasetSnapshot> {
  const db = await getDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DATASETS, "readonly");
    const store = tx.objectStore(STORE_DATASETS);
    const req = store.get(id);
    req.onsuccess = () => {
      if (!req.result) {
        reject(new Error("Dataset snapshot not found in local storage"));
      } else {
        resolve(req.result as DatasetSnapshot);
      }
    };
    req.onerror = () => {
      reject(req.error);
    };
    tx.onabort = () => {
      reject(tx.error || new Error("Transaction aborted"));
    };
  });
}

/**
 * Deletes a saved dataset snapshot by ID from the user's browser storage.
 */
export async function deleteDatasetLocally(id: string): Promise<void> {
  const db = await getDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DATASETS, "readwrite");
    const store = tx.objectStore(STORE_DATASETS);
    store.delete(id);
    tx.oncomplete = () => {
      resolve();
    };
    tx.onerror = () => {
      reject(tx.error);
    };
    tx.onabort = () => {
      reject(tx.error || new Error("Transaction aborted"));
    };
  });
}

/**
 * Lists all saved trained models from the user's browser storage.
 */
export async function listLocalModels(): Promise<TrainedModelRecord[]> {
  try {
    const db = await getDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_MODELS, "readonly");
      const store = tx.objectStore(STORE_MODELS);
      const req = store.getAll();
      req.onsuccess = () => {
        const models = ((req.result as TrainedModelRecord[]) || []).sort(
          (a, b) => b.trainedAt - a.trainedAt
        );
        resolve(models);
      };
      req.onerror = () => {
        reject(req.error);
      };
      tx.onabort = () => {
        reject(tx.error || new Error("Transaction aborted"));
      };
    });
  } catch {
    return [];
  }
}

/**
 * Saves a trained model record locally in the user's browser storage.
 */
export async function saveModelLocally(
  modelRecord: TrainedModelRecord
): Promise<TrainedModelRecord> {
  const db = await getDatabase();
  const record: TrainedModelRecord = {
    ...modelRecord,
    id:
      modelRecord.id ||
      `mdl_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    trainedAt: modelRecord.trainedAt || Date.now(),
  };

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_MODELS, "readwrite");
    const store = tx.objectStore(STORE_MODELS);
    store.put(record);
    tx.oncomplete = () => {
      resolve(record);
    };
    tx.onerror = () => {
      reject(tx.error);
    };
    tx.onabort = () => {
      reject(tx.error || new Error("Transaction aborted"));
    };
  });
}

/**
 * Deletes a saved model from the user's browser storage.
 */
export async function deleteModelLocally(id: string): Promise<void> {
  const db = await getDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_MODELS, "readwrite");
    const store = tx.objectStore(STORE_MODELS);
    store.delete(id);
    tx.oncomplete = () => {
      resolve();
    };
    tx.onerror = () => {
      reject(tx.error);
    };
    tx.onabort = () => {
      reject(tx.error || new Error("Transaction aborted"));
    };
  });
}
