"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CameraOff,
  Eye,
  EyeOff,
  FlipHorizontal,
  Hand,
  Loader2,
  Play,
  RefreshCw,
  Settings2,
  Sparkles,
  Square,
  Timer,
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
  const [minConfidence, setMinConfidence] = useState<number>(0.4);
  const [showSettings, setShowSettings] = useState<boolean>(false);

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

        const candidates: { path: string; delegate: "GPU" | "CPU" }[] = [
          { path: "/mediapipe/models/hand_landmarker.task", delegate: "GPU" },
          { path: "/mediapipe/models/hand_landmarker.task", delegate: "CPU" },
          {
            path: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          {
            path: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "CPU",
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
              minHandPresenceConfidence: minConfidence,
              minTrackingConfidence: minConfidence,
            });
            activeCanvas = helperCanvas;
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
  }, [onModelStatusChange, retryCount, maxHands, minConfidence]);

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
            // In Safari/WebKit, passing HTMLVideoElement directly to WebGL texImage2D
            // can fail or stall due to hardware video decoder synchronization.
            // Drawing the video frame to an offscreen 2D canvas guarantees clean pixel data across all browsers.
            if (!videoFrameCanvasRef.current) {
              videoFrameCanvasRef.current = document.createElement("canvas");
            }
            const vfc = videoFrameCanvasRef.current;
            if (vfc.width !== vw || vfc.height !== vh) {
              vfc.width = vw;
              vfc.height = vh;
            }
            const vfCtx = vfc.getContext("2d");
            let inputSource: TexImageSource = video;
            if (vfCtx) {
              vfCtx.drawImage(video, 0, 0, vw, vh);
              inputSource = vfc;
            }

            const result = landmarker.detectForVideo(inputSource, nowMs);
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
            console.warn("Detection frame skipped:", err);
          }
        }
      } else if (canvas) {
        if (cameraOn && video && video.paused) {
          video.play().catch(() => {});
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
      setCaptureNotice("Select or create a gesture class first!");
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
    <div className="flex flex-col rounded-2xl border border-slate-800/90 bg-slate-900/70 shadow-xl overflow-hidden">
      {/* Stage Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 bg-slate-900/95 px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <span
            className={`flex h-2.5 w-2.5 rounded-full ${
              !cameraOn
                ? "bg-slate-600"
                : handDetected
                ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
                : "bg-amber-400"
            }`}
          />
          <span className="text-xs font-semibold text-slate-200">
            {!cameraOn
              ? "Camera Paused"
              : handDetected
              ? `${liveHands.length} Hand${liveHands.length > 1 ? "s" : ""} Tracked (${liveHands
                  .map((h) => `${h.handedness} ${(h.score * 100).toFixed(0)}%`)
                  .join(", ")})`
              : "Show your hand to the camera"}
          </span>
        </div>

        {/* Stage Quick Toggles */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setMirrored((m) => !m)}
            title="Mirror Video Feed"
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
              mirrored
                ? "bg-slate-800 text-cyan-300 border border-cyan-500/30"
                : "bg-slate-800/50 text-slate-400 hover:text-slate-200"
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
                ? "bg-slate-800 text-cyan-300 border border-cyan-500/30"
                : "bg-slate-800/50 text-slate-400 hover:text-slate-200"
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
            title="Tracking Settings"
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
              showSettings
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                : "bg-slate-800/50 text-slate-400 hover:text-slate-200"
            }`}
          >
            <Settings2 className="h-3.5 w-3.5" />
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
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 bg-slate-950/80 px-4 py-2.5 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Max Hands:</span>
            <div className="flex rounded-lg border border-slate-800 bg-slate-900 p-0.5">
              {[1, 2].map((n) => (
                <button
                  key={n}
                  onClick={() => setMaxHands(n as 1 | 2)}
                  className={`rounded-md px-2.5 py-0.5 font-medium transition ${
                    maxHands === n
                      ? "bg-cyan-500 text-slate-950"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {n} {n === 1 ? "Hand" : "Hands"}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-slate-400">
              Min Confidence: {(minConfidence * 100).toFixed(0)}%
            </span>
            <input
              type="range"
              min={0.25}
              max={0.8}
              step={0.05}
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="h-1.5 w-20 accent-cyan-400 cursor-pointer"
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-slate-400">Burst:</span>
            <select
              value={burstTarget}
              onChange={(e) => setBurstTarget(Number(e.target.value))}
              className="rounded-md border border-slate-700 bg-slate-900 px-2 py-0.5 text-slate-200"
            >
              <option value={10}>10 photos</option>
              <option value={15}>15 photos</option>
              <option value={25}>25 photos</option>
            </select>
          </div>
        </div>
      )}

      {/* Video & Canvas Stage */}
      <div className="relative aspect-[4/3] w-full bg-slate-950 flex items-center justify-center overflow-hidden">
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
              className="mt-1 flex items-center gap-1.5 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 transition hover:bg-cyan-400"
            >
              <Play className="h-3.5 w-3.5" />
              <span>Start Camera</span>
            </button>
          </div>
        )}

        {/* Active Recording Class Badge Overlay (Top-Left) */}
        {activeClass && cameraOn && (
          <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-2 rounded-lg border border-slate-700/80 bg-slate-950/85 px-3 py-1.5 backdrop-blur-md">
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
          <div className="pointer-events-none absolute bottom-3 left-3 right-3 flex flex-col gap-2 rounded-xl border border-cyan-500/40 bg-slate-950/90 p-3 shadow-2xl backdrop-blur-md sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-300">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-wider text-cyan-400">
                  Live Prediction
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-extrabold text-white">
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
                  <span className="w-20 truncate text-right font-medium text-slate-300">
                    {item.className}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full transition-all duration-150"
                      style={{
                        width: `${Math.max(2, item.probability * 100)}%`,
                        backgroundColor: item.color,
                      }}
                    />
                  </div>
                  <span className="w-9 font-mono text-[10px] text-slate-400">
                    {(item.probability * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Capture Action Deck */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800/80 bg-slate-900/95 px-4 py-3">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Hand className="h-4 w-4 text-cyan-400 shrink-0" />
          <span>
            Shortcut: press{" "}
            <kbd className="rounded border border-slate-700 bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] text-slate-200">
              Space
            </kbd>{" "}
            to capture
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={captureNow}
            disabled={!cameraOn || !activeClass}
            className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-4 py-2 text-xs font-bold text-white shadow-md shadow-cyan-500/20 transition hover:from-cyan-400 hover:to-blue-500 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Camera className="h-4 w-4" />
            <span>Capture Photo</span>
          </button>

          <button
            onClick={startTimerCapture}
            disabled={!cameraOn || !activeClass || countdown !== null}
            title="3-second countdown before capturing"
            className="flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-slate-600 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Timer className="h-4 w-4 text-amber-400" />
            <span>3s Timer</span>
          </button>

          <button
            onClick={toggleBurstCapture}
            disabled={!cameraOn || !activeClass}
            title={`Automatically capture ${burstTarget} frames while your hand is in view`}
            className={`flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-semibold transition ${
              isBursting
                ? "border border-rose-500/50 bg-rose-500/20 text-rose-300"
                : "border border-amber-500/30 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20"
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
                <Zap className="h-4 w-4" />
                <span>Auto-Burst ({burstTarget})</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
