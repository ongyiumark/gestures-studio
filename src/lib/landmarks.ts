import { DetectedHand, Point3D } from "@/types/handpose";

export const LANDMARK_NAMES = [
  "wrist",
  "thumb_cmc",
  "thumb_mcp",
  "thumb_ip",
  "thumb_tip",
  "index_finger_mcp",
  "index_finger_pip",
  "index_finger_dip",
  "index_finger_tip",
  "middle_finger_mcp",
  "middle_finger_pip",
  "middle_finger_dip",
  "middle_finger_tip",
  "ring_finger_mcp",
  "ring_finger_pip",
  "ring_finger_dip",
  "ring_finger_tip",
  "pinky_mcp",
  "pinky_pip",
  "pinky_dip",
  "pinky_tip",
] as const;

export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  // Index finger
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  // Middle finger
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  // Ring finger
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  // Pinky
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  // Palm knuckles
  [5, 9],
  [9, 13],
  [13, 17],
];

export const CLASS_COLORS = [
  "#38bdf8", // sky-400
  "#a78bfa", // violet-400
  "#34d399", // emerald-400
  "#fbbf24", // amber-400
  "#f472b6", // pink-400
  "#fb7185", // rose-400
  "#60a5fa", // blue-400
  "#4ade80", // green-400
  "#f97316", // orange-500
  "#2dd4bf", // teal-400
];

export function getClassColor(index: number): string {
  return CLASS_COLORS[index % CLASS_COLORS.length];
}

export interface DrawHandOptions {
  compact?: boolean;
  showLabels?: boolean;
  accentColor?: string;
}

/**
 * Draws detected MediaPipe hands on a 2D canvas context.
 * Landmarks are normalized in [0, 1].
 */
export function drawHandsOnCanvas(
  hands: DetectedHand[],
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  options: DrawHandOptions = {}
) {
  ctx.clearRect(0, 0, width, height);
  if (!hands || hands.length === 0) return;

  const { compact = false, showLabels = true, accentColor } = options;

  hands.forEach((hand, handIdx) => {
    const pts = hand.landmarks;
    if (!pts || pts.length < 21) return;

    const primaryColor =
      accentColor ||
      (hand.handedness === "Right" ? "#22d3ee" : "#a855f7");
    const jointColor =
      hand.handedness === "Right" ? "#f43f5e" : "#ec4899";

    // Draw skeleton connections
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const p0 = pts[startIdx];
      const p1 = pts[endIdx];
      if (!p0 || !p1) continue;

      const x0 = p0.x * width;
      const y0 = p0.y * height;
      const x1 = p1.x * width;
      const y1 = p1.y * height;

      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = compact ? 1.5 : 3;
      if (!compact) {
        ctx.shadowColor = primaryColor;
        ctx.shadowBlur = 8;
      }
      ctx.stroke();
    }

    // Draw 21 landmark joints
    pts.forEach((pt, idx) => {
      const x = pt.x * width;
      const y = pt.y * height;
      const isTip = [4, 8, 12, 16, 20].includes(idx);
      const isWrist = idx === 0;

      const radius = compact
        ? isTip
          ? 2.5
          : 1.8
        : isTip
        ? 5.5
        : isWrist
        ? 6
        : 4;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isTip ? "#ffffff" : isWrist ? "#fbbf24" : jointColor;
      ctx.fill();

      if (!compact) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = primaryColor;
        ctx.stroke();
      }
    });

    // Optional handedness + score badge near wrist
    if (showLabels && !compact) {
      const wrist = pts[0];
      const wx = Math.min(Math.max(wrist.x * width, 40), width - 100);
      const wy = Math.min(Math.max(wrist.y * height + 22, 24), height - 12);
      const label = `${hand.handedness} ${(hand.score * 100).toFixed(0)}%`;

      ctx.shadowBlur = 0;
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
      ctx.beginPath();
      ctx.roundRect(wx - 8, wy - 14, textWidth + 16, 20, 6);
      ctx.fill();
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, wx, wy);
    }

    ctx.restore();
  });
}

/**
 * Extracts a 63-dimensional translation- and scale-invariant feature vector
 * from the primary detected hand's 21 3D landmarks.
 *
 * Step 1: Subtract wrist (landmark 0) coordinates -> translation invariance.
 * Step 2: Divide by max Euclidean distance from wrist -> scale invariance.
 * Step 3: Flatten 21 * (dx, dy, dz) = 63 features.
 */
export function extractFeatureVector(hands: DetectedHand[]): number[] | null {
  if (!hands || hands.length === 0) return null;
  const primary = hands[0];
  if (!primary.landmarks || primary.landmarks.length < 21) return null;

  const pts = primary.landmarks;
  const wrist = pts[0];

  const centered: Point3D[] = pts.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));

  let maxDist = 0;
  for (const p of centered) {
    const dist = Math.hypot(p.x, p.y, p.z);
    if (dist > maxDist) maxDist = dist;
  }

  const scale = maxDist > 1e-6 ? maxDist : 1;

  const features: number[] = [];
  for (const p of centered) {
    features.push(p.x / scale, p.y / scale, p.z / scale);
  }

  return features;
}

/**
 * Creates a slightly perturbed copy of a 63-D feature vector (jitter + minor 2D rotation)
 * to improve classifier generalization on small user-collected datasets.
 */
export function augmentFeatureVector(
  features: number[],
  noiseStd = 0.018,
  maxAngleRad = 0.14
): number[] {
  const angle = (Math.random() * 2 - 1) * maxAngleRad;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  const out: number[] = new Array(features.length);
  for (let i = 0; i < features.length; i += 3) {
    const x = features[i];
    const y = features[i + 1];
    const z = features[i + 2];

    const rx = x * cosA - y * sinA + gaussianRandom() * noiseStd;
    const ry = x * sinA + y * cosA + gaussianRandom() * noiseStd;
    const rz = z + gaussianRandom() * noiseStd;

    out[i] = rx;
    out[i + 1] = ry;
    out[i + 2] = rz;
  }
  return out;
}

function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Converts MediaPipe DetectedHand[] into a comprehensive JSON structure that
 * includes both modern MediaPipe fields AND legacy @tensorflow-models/handpose
 * compatible pixel landmarks + finger annotations.
 */
export function formatHandAnnotationsForExport(
  hands: DetectedHand[],
  width: number,
  height: number
) {
  return hands.map((hand) => {
    const pixelLandmarks = hand.landmarks.map((pt) => [
      Number((pt.x * width).toFixed(2)),
      Number((pt.y * height).toFixed(2)),
      Number((pt.z * width).toFixed(2)),
    ]);

    return {
      handedness: hand.handedness,
      handInViewConfidence: hand.score,
      normalizedLandmarks: hand.landmarks,
      worldLandmarks: hand.worldLandmarks,
      landmarks: pixelLandmarks,
      annotations: {
        palmBase: [pixelLandmarks[0]],
        thumb: pixelLandmarks.slice(1, 5),
        indexFinger: pixelLandmarks.slice(5, 9),
        middleFinger: pixelLandmarks.slice(9, 13),
        ringFinger: pixelLandmarks.slice(13, 17),
        pinky: pixelLandmarks.slice(17, 21),
      },
    };
  });
}
