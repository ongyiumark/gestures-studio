"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CameraOff,
  Eye,
  EyeOff,
  FlipHorizontal,
  Hand,
  HelpCircle,
  Info,
  Loader2,
  Play,
  RefreshCw,
  Settings2,
  Sparkles,
  Square,
  Timer,
  X,
  Zap,
} from "lucide-react";
import {
  CapturedSample,
  DetectedHand,
  PoseClassMeta,
  PredictionResult,
} from "@/types/handpose";
import { drawHandsOnCanvas } from "@/lib/landmarks";

interface CameraStageProps {
  activeClass: PoseClassMeta | null;
  onCaptureSample: (sample: CapturedSample) => void;
  onHandsDetected: (hands: DetectedHand[], fps: number) => void;
  onModelStatusChange: (ready: boolean) => void;
  livePrediction: PredictionResult | null;
  classifierActive: boolean;
}

export default function CameraStage({
  activeClass,
  onCaptureSample,
  onHandsDetected,
  onModelStatusChange,
  livePrediction,
  classifierActive,
}: CameraStageProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const activeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoFrameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const lastInferenceTimeRef = useRef<number>(0);
  const lastStateUpdateRef = useRef<number>(0);
  const fpsTimestampsRef = useRef<number[]>([]);
  const latestHandsRef = useRef<DetectedHand[]>([]);

  const [cameraOn, setCameraOn] = useState<boolean>(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [mirrored, setMirrored] = useState<boolean>(true);
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true);
  const [maxHands, setMaxHands] = useState<1 | 2>(2);
  const [minConfidence, setMinConfidence] = useState<number>(0.35);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [showConfidenceTooltip, setShowConfidenceTooltip] = useState<boolean>(false);

  const [activeDelegate, setActiveDelegate] = useState<"CPU" | "GPU">("CPU");
  const [videoStatus, setVideoStatus] = useState<string>("Connecting…");
  const [videoPaused, setVideoPaused] = useState<boolean>(false);
  const [lastDetectError, setLastDetectError] = useState<string | null>(null);
  const [fpsVal, setFpsVal] = useState<number>(0);

  const [modelReady, setModelReady] = useState<boolean>(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState<number>(0);
  const [liveHands, setLiveHands] = useState<DetectedHand[]>([]);

  // Capture modes: single, timer countdown, or auto-burst
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isBursting, setIsBursting] = useState<boolean>(false);
  const [burstTarget, setBurstTarget] = useState<number>(15);
  const [burstCount, setBurstCount] = useState<number>(0);
  const [flashEffect, setFlashEffect] = useState<boolean>(false);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);

  // Initialize MediaPipe HandLandmarker using CPU XNNPACK (SIMD)
  useEffect(() => {
    let cancelled = false;

    // Filter out Emscripten stderr info messages that Next.js dev server treats as console errors
    if (typeof window !== "undefined" && !(window as unknown as { __mpStderrPatched?: boolean }).__mpStderrPatched) {
      (window as unknown as { __mpStderrPatched?: boolean }).__mpStderrPatched = true;
      const originalConsoleError = console.error;
      console.error = (...args: unknown[]) => {
        const text = typeof args[0] === "string" ? args[0] : "";
        if (
          text.includes("INFO: Created TensorFlow Lite XNNPACK delegate for CPU") ||
          text.startsWith("INFO: ") ||
          text.includes("gl_context.cc") ||
          text.includes("GL version mismatch") ||
          text.includes("Error querying for GL extensions") ||
          text.includes("Aborted()")
        ) {
          console.info(...args);
          return;
        }
        originalConsoleError.apply(console, args);
      };
    }

    async function initMediaPipe() {
      try {
        setModelError(null);
        setModelReady(false);
        const vision = await import("@mediapipe/tasks-vision");
        const { FilesetResolver, HandLandmarker } = vision;

        // Try local self-hosted WASM first, fallback to CDN
        let wasmFileset;
        try {
          wasmFileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        } catch {
          wasmFileset = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
          );
        }

        const preferred = activeDelegate;
        const fallback = preferred === "CPU" ? "GPU" : "CPU";
        const candidates: { path: string; delegate: "CPU" | "GPU" }[] = [
          { path: "/mediapipe/models/hand_landmarker.task", delegate: preferred },
          { path: "/mediapipe/models/hand_landmarker.task", delegate: fallback },
          {
            path: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: preferred,
          },
          {
            path: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: fallback,
          },
        ];

        let landmarker = null;
        let lastError: unknown = null;
        let activeCanvas: HTMLCanvasElement | null = null;

        for (const candidate of candidates) {
          if (cancelled) return;
          const helperCanvas = document.createElement("canvas");
          helperCanvas.width = 640;
          helperCanvas.height = 480;
          helperCanvas.style.position = "fixed";
          helperCanvas.style.top = "-9999px";
          helperCanvas.style.left = "-9999px";
          helperCanvas.style.width = "640px";
          helperCanvas.style.height = "480px";
          helperCanvas.style.opacity = "0";
          helperCanvas.style.pointerEvents = "none";
          helperCanvas.style.visibility = "visible";
          helperCanvas.setAttribute("aria-hidden", "true");
          document.body.appendChild(helperCanvas);

          try {
            landmarker = await HandLandmarker.createFromOptions(wasmFileset, {
              baseOptions: {
                modelAssetPath: candidate.path,
                delegate: candidate.delegate,
              },
              canvas: helperCanvas,
              runningMode: "VIDEO",
              numHands: maxHands,
              minHandDetectionConfidence: minConfidence,
              // Lower presence and tracking thresholds allow MediaPipe to keep tracking both hands
              // even when they touch, overlap, or partially occlude each other in multi-hand poses.
              minHandPresenceConfidence: Math.max(0.18, minConfidence - 0.15),
              minTrackingConfidence: Math.max(0.18, minConfidence - 0.15),
            });
            activeCanvas = helperCanvas;
            setActiveDelegate(candidate.delegate);
            break;
          } catch (err) {
            lastError = err;
            helperCanvas.remove();
            console.warn(
              `HandLandmarker init attempt failed (${candidate.path}, ${candidate.delegate}):`,
              err
            );
          }
        }

        if (!landmarker) {
          throw lastError ?? new Error("All HandLandmarker init attempts failed");
        }

        if (cancelled) {
          landmarker.close();
          return;
        }

        landmarkerRef.current = landmarker;
        activeCanvasRef.current = activeCanvas;
        setModelReady(true);
        onModelStatusChange(true);
      } catch (err) {
        console.error("Failed to initialize MediaPipe HandLandmarker:", err);
        if (!cancelled) {
          const detail =
            err instanceof Error ? err.message : "Unknown initialization error";
          setModelError(`Could not load MediaPipe HandLandmarker (${detail}).`);
          onModelStatusChange(false);
        }
      }
    }

    initMediaPipe();

    return () => {
      cancelled = true;
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
        landmarkerRef.current = null;
      }
      if (activeCanvasRef.current) {
        activeCanvasRef.current.remove();
        activeCanvasRef.current = null;
      }
    };
  }, [onModelStatusChange, retryCount, maxHands, minConfidence, activeDelegate]);

  // Start / Stop Webcam Stream
  useEffect(() => {
    let activeStream: MediaStream | null = null;
    let isMounted = true;

    async function setupStream() {
      if (!cameraOn) {
        if (videoRef.current && videoRef.current.srcObject) {
          const s = videoRef.current.srcObject as MediaStream;
          s.getTracks().forEach((t) => t.stop());
          videoRef.current.srcObject = null;
        }
        setLiveHands([]);
        latestHandsRef.current = [];
        return;
      }

      try {
        setCameraError(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          },
          audio: false,
        });

        if (!isMounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.muted = true;
          videoRef.current.defaultMuted = true;
          videoRef.current.setAttribute("playsinline", "true");
          videoRef.current.setAttribute("webkit-playsinline", "true");
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch((err) => {
            console.warn("Video playback autostart pending user action:", err);
          });
        }
      } catch (err) {
        console.warn("Webcam access error:", err);
        if (isMounted) {
          setCameraError(
            "Camera access unavailable or denied. Check browser permissions."
          );
          setCameraOn(false);
        }
      }
    }

    setupStream();

    return () => {
      isMounted = false;
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [cameraOn]);

  // Real-time requestAnimationFrame detection loop
  useEffect(() => {
    let running = true;

    const detectFrame = () => {
      if (!running) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const landmarker = landmarkerRef.current;

      if (
        cameraOn &&
        video &&
        canvas &&
        landmarker &&
        video.readyState >= 2 &&
        video.videoWidth > 0
      ) {
        const vw = video.videoWidth;
        const vh = video.videoHeight;

        if (canvas.width !== vw || canvas.height !== vh) {
          canvas.width = vw;
          canvas.height = vh;
        }

        const nowMs = performance.now();
        // Run inference at ~30 FPS with monotonically increasing timestamp
        if (nowMs - lastInferenceTimeRef.current >= 30) {
          lastInferenceTimeRef.current = nowMs;

          try {
            let result;
            try {
              result = landmarker.detectForVideo(video, Math.round(nowMs));
            } catch {
              if (!videoFrameCanvasRef.current) {
                videoFrameCanvasRef.current = document.createElement("canvas");
              }
              const vfc = videoFrameCanvasRef.current;
              if (vfc.width !== vw || vfc.height !== vh) {
                vfc.width = vw;
                vfc.height = vh;
              }
              const vfCtx = vfc.getContext("2d");
              if (vfCtx) {
                vfCtx.drawImage(video, 0, 0, vw, vh);
                result = landmarker.detectForVideo(vfc, Math.round(nowMs));
              }
            }

            if (!result) return;

            const handsList = result.landmarks || [];
            const handednessList = result.handedness || result.handednesses || [];

            const detected: DetectedHand[] = handsList.map(
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (pts: any[], idx: number) => {
                const handInfo = handednessList[idx]?.[0];
                const rawHandedness = handInfo?.categoryName || "Right";
                const handedness: "Left" | "Right" =
                  rawHandedness === "Left" ? "Left" : "Right";
                const score = handInfo?.score ?? 0.85;
                const worldPts = result.worldLandmarks?.[idx] || pts;

                return {
                  handedness,
                  score,
                  landmarks: pts.map((p) => ({ x: p.x, y: p.y, z: p.z })),
                  worldLandmarks: worldPts.map(
                    (p: { x: number; y: number; z: number }) => ({
                      x: p.x,
                      y: p.y,
                      z: p.z,
                    })
                  ),
                };
              }
            );

            latestHandsRef.current = detected;

            // Rolling FPS telemetry
            const timestamps = fpsTimestampsRef.current;
            timestamps.push(nowMs);
            while (timestamps.length > 0 && timestamps[0] < nowMs - 1000) {
              timestamps.shift();
            }
            const currentFps = timestamps.length;

            onHandsDetected(detected, currentFps);
            setFpsVal(currentFps);
            setVideoStatus(`${vw}×${vh}`);
            if (videoPaused) setVideoPaused(false);
            if (lastDetectError) setLastDetectError(null);

            // Throttle React state update to 10 FPS to keep main thread fast
            if (nowMs - lastStateUpdateRef.current >= 100) {
              lastStateUpdateRef.current = nowMs;
              setLiveHands(detected);
            }

            // Real-time canvas skeleton rendering (60 FPS, direct 2D context)
            const ctx = canvas.getContext("2d");
            if (ctx) {
              if (showSkeleton) {
                drawHandsOnCanvas(detected, ctx, vw, vh, {
                  compact: false,
                  showLabels: true,
                  accentColor: activeClass?.color,
                });
              } else {
                ctx.clearRect(0, 0, vw, vh);
              }
            }
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            setLastDetectError(msg);
            console.warn("Detection frame skipped:", err);
          }
        }
      } else if (canvas) {
        if (cameraOn && video) {
          if (video.paused) {
            setVideoPaused(true);
            video.play().catch(() => {});
          } else if (video.readyState < 2) {
            setVideoStatus(`Connecting (ready: ${video.readyState})`);
          }
        }
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }

      rafRef.current = requestAnimationFrame(detectFrame);
    };

    rafRef.current = requestAnimationFrame(detectFrame);

    return () => {
      running = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [cameraOn, showSkeleton, activeClass?.color, onHandsDetected]);

  // Capture a single frame + current 21-keypoint hand landmarks
  const captureNow = useCallback((): boolean => {
    if (!activeClass) {
      setCaptureNotice("Select or create a hand pose class first!");
      setTimeout(() => setCaptureNotice(null), 2200);
      return false;
    }

    const video = videoRef.current;
    const hands = latestHandsRef.current;

    if (!video || !cameraOn || video.readyState < 2) {
      setCaptureNotice("Camera must be active to capture samples.");
      setTimeout(() => setCaptureNotice(null), 2200);
      return false;
    }

    if (!hands || hands.length === 0) {
      setCaptureNotice("Hold your hand clearly in front of the camera.");
      setTimeout(() => setCaptureNotice(null), 1800);
      return false;
    }

    const tempCanvas = document.createElement("canvas");
    const targetW = 320;
    const targetH =
      Math.round((video.videoHeight / (video.videoWidth || 640)) * targetW) ||
      240;
    tempCanvas.width = targetW;
    tempCanvas.height = targetH;
    const ctx = tempCanvas.getContext("2d");
    if (!ctx) return false;

    ctx.drawImage(video, 0, 0, targetW, targetH);
    const imageDataUrl = tempCanvas.toDataURL("image/jpeg", 0.82);

    const sample: CapturedSample = {
      id: `smp_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      className: activeClass.name,
      timestamp: Date.now(),
      imageDataUrl,
      width: video.videoWidth || 640,
      height: video.videoHeight || 480,
      hands: JSON.parse(JSON.stringify(hands)),
    };

    onCaptureSample(sample);
    setFlashEffect(true);
    setTimeout(() => setFlashEffect(false), 120);
    return true;
  }, [activeClass, cameraOn, onCaptureSample]);

  // Spacebar shortcut for quick single capture
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        captureNow();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [captureNow]);

  // 3-Second Countdown Timer Capture
  const startTimerCapture = () => {
    if (countdown !== null || isBursting) return;
    setCountdown(3);
  };

  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) {
      setCountdown(null);
      captureNow();
      return;
    }
    const timer = setTimeout(() => {
      setCountdown((prev) => (prev !== null ? prev - 1 : null));
    }, 1000);
    return () => clearTimeout(timer);
  }, [countdown, captureNow]);

  // Auto-Burst Capture Loop
  const toggleBurstCapture = () => {
    if (isBursting) {
      setIsBursting(false);
      setBurstCount(0);
    } else {
      if (!activeClass) {
        setCaptureNotice("Select a class before starting Burst Capture.");
        setTimeout(() => setCaptureNotice(null), 2000);
        return;
      }
      setBurstCount(0);
      setIsBursting(true);
    }
  };

  useEffect(() => {
    if (!isBursting) return;
    const interval = setInterval(() => {
      if (latestHandsRef.current.length > 0) {
        const ok = captureNow();
        if (ok) {
          setBurstCount((prev) => {
            const next = prev + 1;
            if (next >= burstTarget) {
              setIsBursting(false);
              return 0;
            }
            return next;
          });
        }
      }
    }, 220);

    return () => clearInterval(interval);
  }, [isBursting, burstTarget, captureNow]);

  const handDetected = liveHands.length > 0;

  return (
    <div className="flex flex-col rounded-2xl border border-ocean-700/50 bg-[#101e2e] shadow-2xl">
      {/* Stage Top Bar */}
      <div className="flex items-center justify-between gap-2 border-b border-ocean-700/40 bg-gradient-to-r from-[#112437] to-[#152e46] px-3.5 py-2.5 sm:px-4 sm:py-3 rounded-t-2xl min-h-[50px]">
        <div className="flex items-center gap-2 sm:gap-2.5 min-w-0 flex-1">
          <span
            className={`flex h-2.5 w-2.5 shrink-0 rounded-full ${
              !cameraOn
                ? "bg-slate-600"
                : handDetected
                ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
                : "bg-amber-400"
            }`}
          />
          <span className="text-xs font-semibold text-slate-100 truncate">
            {!cameraOn
              ? "Camera Paused"
              : handDetected
              ? `${liveHands.length === 1 ? "1 Hand" : `${liveHands.length} Hands`} Tracked (${liveHands
                  .map((h) => `${h.handedness[0]}: ${(h.score * 100).toFixed(0)}%`)
                  .join(", ")})`
              : "Show hand(s) to the camera"}
          </span>
        </div>

        {/* Stage Quick Toggles */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={() => setMirrored((m) => !m)}
            title="Mirror Video Feed"
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
              mirrored
                ? "bg-[#183959] text-sky-200 border border-ocean-500/50"
                : "bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-transparent"
            }`}
          >
            <FlipHorizontal className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Mirror</span>
          </button>

          <button
            onClick={() => setShowSkeleton((s) => !s)}
            title="Toggle 21-Keypoint Skeleton Overlay"
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
              showSkeleton
                ? "bg-[#183959] text-sky-200 border border-ocean-500/50"
                : "bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-transparent"
            }`}
          >
            {showSkeleton ? (
              <Eye className="h-3.5 w-3.5" />
            ) : (
              <EyeOff className="h-3.5 w-3.5" />
            )}
            <span className="hidden sm:inline">Skeleton</span>
          </button>

          <button
            onClick={() => setShowSettings((s) => !s)}
            title="Tracking Settings (Confidence, Engine, Hands)"
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
              showSettings
                ? "bg-amber-500/20 text-gold-200 border border-amber-500/40"
                : "bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-transparent"
            }`}
          >
            <Settings2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Controls</span>
          </button>

          <button
            onClick={() => setCameraOn((c) => !c)}
            title={cameraOn ? "Turn Off Camera" : "Turn On Camera"}
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
              cameraOn
                ? "bg-rose-500/15 text-rose-300 border border-rose-500/30 hover:bg-rose-500/25"
                : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/30"
            }`}
          >
            {cameraOn ? (
              <>
                <CameraOff className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Stop</span>
              </>
            ) : (
              <>
                <Camera className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Start</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Collapsible Tracking Settings Bar */}
      {showSettings && (
        <div className="relative flex flex-wrap items-center justify-between gap-y-2.5 gap-x-4 border-b border-ocean-700/40 bg-[#0c1825]/95 px-3.5 py-2.5 text-xs text-slate-300 sm:px-4">
          <div className="flex items-center gap-2">
            <span className="text-slate-300">Max Hands:</span>
            <div className="flex rounded-lg border border-ocean-700/60 bg-[#08111a] p-0.5">
              {[1, 2].map((n) => (
                <button
                  key={n}
                  onClick={() => setMaxHands(n as 1 | 2)}
                  className={`rounded-md px-2.5 py-0.5 font-medium transition ${
                    maxHands === n
                      ? "bg-ocean-600 text-white font-semibold shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {n} {n === 1 ? "Hand" : "Hands"}
                </button>
              ))}
            </div>
          </div>

          {/* Min Confidence with Interactive Tooltip Popover */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-slate-300">
                Min Confidence: {(minConfidence * 100).toFixed(0)}%
              </span>

              {/* Tooltip trigger wrapped in relative container */}
              <div className="relative inline-flex items-center">
                <button
                  type="button"
                  onClick={() => setShowConfidenceTooltip((v) => !v)}
                  onMouseEnter={() => setShowConfidenceTooltip(true)}
                  onMouseLeave={() => setShowConfidenceTooltip(false)}
                  className="rounded-full p-1 text-sky-400 hover:text-gold-200 focus:outline-none transition"
                  aria-label="Explain Min Confidence"
                >
                  <HelpCircle className="h-4 w-4" />
                </button>

                {/* Accessible Info Popover Tooltip (Centered on button, never clipped) */}
                {showConfidenceTooltip && (
                  <div
                    className="absolute top-full left-1/2 -translate-x-1/2 z-50 mt-2 w-72 sm:w-80 max-w-[calc(100vw-40px)] rounded-xl border border-ocean-600/70 bg-[#0e2135] p-3.5 text-xs text-slate-200 shadow-2xl backdrop-blur-md pointer-events-auto"
                    onMouseEnter={() => setShowConfidenceTooltip(true)}
                    onMouseLeave={() => setShowConfidenceTooltip(false)}
                  >
                    <div className="flex items-center justify-between pb-1.5 border-b border-ocean-700/50 mb-2">
                      <span className="font-bold text-white flex items-center gap-1.5">
                        <Info className="h-3.5 w-3.5 text-gold-200" />
                        Min Confidence ({(minConfidence * 100).toFixed(0)}%)
                      </span>
                      <button
                        onClick={() => setShowConfidenceTooltip(false)}
                        className="rounded p-0.5 text-slate-400 hover:text-white"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                    <p className="text-[11px] leading-relaxed text-slate-300">
                      Sets how strictly MediaPipe requires a hand match before plotting 21 3D landmarks:
                    </p>
                    <div className="mt-2 space-y-1.5 text-[11px]">
                      <div className="rounded-lg bg-[#08121d] p-2 border border-ocean-800">
                        <strong className="text-sky-300">Lower (25%–35%):</strong> Easier detection in dim light, rapid motion, or partial angles, with slight landmark jitter.
                      </div>
                      <div className="rounded-lg bg-[#08121d] p-2 border border-ocean-800">
                        <strong className="text-amber-300">Higher (50%–70%):</strong> Stricter detection; requires a crisp, well-lit hand, filtering out background false-positives.
                      </div>
                    </div>
                    <p className="mt-2 text-[10px] text-sky-200/80">
                      Default <strong>35%–40%</strong> is recommended for fluid, responsive studio capture.
                    </p>
                  </div>
                )}
              </div>
            </div>

            <input
              type="range"
              min={0.25}
              max={0.8}
              step={0.05}
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="h-1.5 w-20 accent-terracotta-500 cursor-pointer"
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-slate-300">Burst:</span>
            <select
              value={burstTarget}
              onChange={(e) => setBurstTarget(Number(e.target.value))}
              className="rounded-md border border-ocean-700/60 bg-[#08111a] px-2 py-0.5 text-slate-100"
            >
              <option value={10}>10 photos</option>
              <option value={15}>15 photos</option>
              <option value={25}>25 photos</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-slate-300">Engine:</span>
            <div className="flex rounded-lg border border-ocean-700/60 bg-[#08111a] p-0.5">
              {(["CPU", "GPU"] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => {
                    setActiveDelegate(d);
                    setRetryCount((r) => r + 1);
                  }}
                  className={`rounded-md px-2.5 py-0.5 font-medium transition ${
                    activeDelegate === d
                      ? "bg-ocean-600 text-white font-semibold shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Video & Canvas Stage */}
      <div className="relative aspect-[4/3] w-full bg-[#08111a] flex items-center justify-center overflow-hidden">
        <video
          ref={videoRef}
          playsInline
          muted
          autoPlay
          onLoadedMetadata={() => {
            if (videoRef.current) {
              videoRef.current.muted = true;
              videoRef.current.play().catch(() => {});
            }
          }}
          onCanPlay={() => {
            if (videoRef.current) {
              videoRef.current.muted = true;
              videoRef.current.play().catch(() => {});
            }
          }}
          className={`h-full w-full object-cover transition-opacity duration-200 ${
            cameraOn ? "opacity-100" : "opacity-0"
          } ${mirrored ? "-scale-x-100" : ""}`}
        />

        <canvas
          ref={canvasRef}
          className={`pointer-events-none absolute inset-0 h-full w-full object-cover ${
            mirrored ? "-scale-x-100" : ""
          }`}
        />

        {/* Live Diagnostics HUD */}
        <div className="absolute top-2.5 left-2.5 z-20 flex flex-col gap-1 rounded-lg border border-ocean-700/60 bg-[#081320]/85 px-2.5 py-1.5 font-mono text-[10px] text-slate-200 backdrop-blur-sm pointer-events-none">
          <div className="flex items-center gap-1.5">
            <span
              className={`h-2 w-2 rounded-full ${
                videoStatus.includes("Active") || videoStatus.includes("×")
                  ? "bg-emerald-400 shadow-[0_0_6px_#34d399]"
                  : "bg-amber-400 animate-pulse"
              }`}
            />
            <span>Cam: {videoStatus}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className={`h-2 w-2 rounded-full ${
                modelReady
                  ? "bg-sky-400 shadow-[0_0_6px_#38bdf8]"
                  : "bg-amber-400"
              }`}
            />
            <span>
              {activeDelegate} • {fpsVal} FPS • {liveHands.length}{" "}
              {liveHands.length === 1 ? "hand" : "hands"}
            </span>
          </div>
          {lastDetectError && (
            <div className="text-rose-400 font-sans text-[9px] max-w-[200px] truncate">
              {lastDetectError}
            </div>
          )}
        </div>

        {/* Safari Autoplay Resume Overlay if blocked */}
        {videoPaused && cameraOn && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-slate-950/80 p-6 text-center backdrop-blur-sm">
            <div className="max-w-xs space-y-3">
              <p className="text-sm font-semibold text-slate-200">
                Webcam stream is paused by your browser.
              </p>
              <button
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.muted = true;
                    videoRef.current.play().catch(() => {});
                    setVideoPaused(false);
                  }
                }}
                className="inline-flex items-center gap-2 rounded-xl bg-cyan-500 px-4 py-2 text-xs font-bold text-slate-950 shadow-lg shadow-cyan-500/25 transition hover:bg-cyan-400 active:scale-95"
              >
                <Play className="h-4 w-4 fill-current" />
                <span>Resume Camera</span>
              </button>
            </div>
          </div>
        )}

        {/* Shutter Flash */}
        {flashEffect && (
          <div className="pointer-events-none absolute inset-0 bg-white/50 transition-opacity" />
        )}

        {/* Countdown Overlay */}
        {countdown !== null && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-slate-950/50 backdrop-blur-[2px]">
            <div className="flex h-28 w-28 items-center justify-center rounded-full border-2 border-cyan-400 bg-slate-900/90 text-5xl font-black text-cyan-300 shadow-2xl">
              {countdown}
            </div>
          </div>
        )}

        {/* Burst Progress Pill */}
        {isBursting && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-2.5 rounded-full border border-amber-500/50 bg-slate-950/90 px-4 py-1.5 text-xs font-semibold text-amber-300 shadow-lg">
            <span className="h-2 w-2 animate-ping rounded-full bg-amber-400" />
            <span>
              Auto-Burst Recording ({burstCount} / {burstTarget}) — Hold pose in
              frame
            </span>
          </div>
        )}

        {/* Transient Capture Notice */}
        {captureNotice && (
          <div className="absolute bottom-16 left-1/2 -translate-x-1/2 rounded-lg border border-rose-500/40 bg-slate-950/95 px-4 py-2 text-xs font-medium text-rose-300 shadow-lg">
            {captureNotice}
          </div>
        )}

        {/* Loading / Camera Off States */}
        {!modelReady && !modelError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/80 p-6 text-center backdrop-blur-sm">
            <Loader2 className="h-8 w-8 animate-spin text-cyan-400" />
            <div>
              <p className="text-sm font-semibold text-white">
                Initializing 3D Hand Tracker…
              </p>
              <p className="mt-1 text-xs text-slate-400">
                Loading MediaPipe WASM runtime &amp; 21-point hand model
              </p>
            </div>
          </div>
        )}

        {modelError && (
          <div className="absolute inset-4 flex flex-col items-center justify-center gap-3 rounded-xl border border-rose-500/30 bg-slate-950/95 p-6 text-center">
            <p className="text-sm font-semibold text-rose-400">{modelError}</p>
            <button
              onClick={() => setRetryCount((c) => c + 1)}
              className="flex items-center gap-1.5 rounded-lg bg-slate-800 px-3.5 py-2 text-xs font-semibold text-white hover:bg-slate-700"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Retry Loading Tracker</span>
            </button>
          </div>
        )}

        {!cameraOn && modelReady && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/90 p-6 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 text-slate-400">
              <CameraOff className="h-7 w-7" />
            </div>
            <div className="max-w-sm">
              <p className="text-sm font-semibold text-slate-200">
                {cameraError ? "Webcam Unavailable" : "Camera Stream Paused"}
              </p>
              <p className="mt-1 text-xs text-slate-400">
                {cameraError ||
                  "Click Start Camera below to resume real-time 3D hand tracking."}
              </p>
            </div>
            <button
              onClick={() => setCameraOn(true)}
              className="mt-1 flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-terracotta-500 to-terracotta-600 px-4 py-2 text-xs font-bold text-white shadow-sm shadow-terracotta-950/40 transition hover:from-terracotta-400 hover:to-terracotta-500"
            >
              <Play className="h-3.5 w-3.5" />
              <span>Start Camera</span>
            </button>
          </div>
        )}

        {/* Active Recording Class Badge Overlay (Top-Left) */}
        {activeClass && cameraOn && (
          <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-2 rounded-lg border border-ocean-700/70 bg-[#0a1624]/90 px-3 py-1.5 backdrop-blur-md">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: activeClass.color }}
            />
            <span className="text-xs text-slate-300">
              Target:{" "}
              <strong className="font-semibold text-white">
                {activeClass.name}
              </strong>
            </span>
          </div>
        )}

        {/* Live Real-Time Prediction HUD Overlay */}
        {classifierActive && livePrediction && handDetected && (
          <div className="pointer-events-none absolute bottom-3 left-3 right-3 flex flex-col gap-2 rounded-xl border border-ocean-400/30 bg-[#091726]/60 p-2.5 sm:p-3 shadow-2xl backdrop-blur-md sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2.5 sm:gap-3">
              <div className="flex h-8 w-8 sm:h-9 sm:w-9 shrink-0 items-center justify-center rounded-lg bg-ocean-500/25 text-sky-300">
                <Sparkles className="h-4 w-4 sm:h-5 sm:w-5" />
              </div>
              <div>
                <div className="text-[9px] sm:text-[10px] font-bold uppercase tracking-wider text-sky-300/90">
                  Live Prediction
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="text-base sm:text-lg font-extrabold text-white drop-shadow-sm">
                    {livePrediction.className}
                  </span>
                  <span className="font-mono text-xs font-bold text-emerald-400">
                    {(livePrediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="flex flex-1 max-w-xs flex-col gap-1">
              {livePrediction.probabilities.slice(0, 3).map((item) => (
                <div
                  key={item.className}
                  className="flex items-center gap-2 text-[11px]"
                >
                  <span className="w-20 truncate text-right font-medium text-slate-200">
                    {item.className}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-900/60 border border-white/5">
                    <div
                      className="h-full rounded-full transition-all duration-150"
                      style={{
                        width: `${Math.max(2, item.probability * 100)}%`,
                        backgroundColor: item.color,
                      }}
                    />
                  </div>
                  <span className="w-9 font-mono text-[10px] text-slate-300">
                    {(item.probability * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Capture Action Deck (Responsive touch-friendly layout) */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-ocean-700/50 bg-gradient-to-r from-[#102235] to-[#142b44] px-4 py-3 rounded-b-2xl">
        <div className="flex items-center gap-2 text-xs text-sky-200/80">
          <Hand className="h-4 w-4 text-terracotta-400 shrink-0" />
          <span>
            Shortcut: press{" "}
            <kbd className="rounded border border-ocean-600/50 bg-[#091522] px-1.5 py-0.5 font-mono text-[11px] text-slate-200">
              Space
            </kbd>{" "}
            to capture
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto">
          <button
            onClick={captureNow}
            disabled={!cameraOn || !activeClass}
            className="flex-1 sm:flex-initial flex items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-terracotta-500 to-terracotta-600 px-4 py-2.5 text-xs font-bold text-white shadow-md shadow-terracotta-950/40 transition hover:from-terracotta-400 hover:to-terracotta-500 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Camera className="h-4 w-4" />
            <span>Capture Photo</span>
          </button>

          <button
            onClick={startTimerCapture}
            disabled={!cameraOn || !activeClass || countdown !== null}
            title="3-second countdown before capturing"
            className="flex-1 sm:flex-initial flex items-center justify-center gap-1.5 rounded-xl border border-ocean-600/60 bg-[#163554]/90 px-3.5 py-2.5 text-xs font-semibold text-slate-100 transition hover:bg-[#1f456c] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Timer className="h-4 w-4 text-gold-300" />
            <span>3s Timer</span>
          </button>

          <button
            onClick={toggleBurstCapture}
            disabled={!cameraOn || !activeClass}
            title={`Automatically capture ${burstTarget} frames while your hand is in view`}
            className={`flex-1 sm:flex-initial flex items-center justify-center gap-1.5 rounded-xl px-3.5 py-2.5 text-xs font-semibold transition ${
              isBursting
                ? "border border-rose-500/50 bg-rose-500/20 text-rose-300"
                : "border border-gold-300/40 bg-gold-200/15 text-gold-100 hover:bg-gold-200/25"
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            {isBursting ? (
              <>
                <Square className="h-3.5 w-3.5 fill-current" />
                <span>
                  Stop ({burstCount}/{burstTarget})
                </span>
              </>
            ) : (
              <>
                <Zap className="h-4 w-4 text-gold-300" />
                <span>Auto-Burst ({burstTarget})</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
