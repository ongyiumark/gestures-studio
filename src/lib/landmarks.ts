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
  "#ea580c", // terracotta-500 (original theme heritage)
  "#38bdf8", // ocean-sky-400
  "#10b981", // emerald-500
  "#f59e0b", // amber-gold-500
  "#8b5cf6", // violet-500
  "#f43f5e", // rose-500
  "#06b6d4", // cyan-500
  "#84cc16", // lime-500
  "#ec4899", // pink-500
  "#6366f1", // indigo-500
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

    const primaryColor = accentColor || "#38bdf8";
    const jointColor = "#f59e0b";

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
 * Total feature dimension for classification.
 * - 63 features: Hand 1 (21 points * 3, wrist-centered & normalized)
 * - 63 features: Hand 2 (21 points * 3, wrist-centered & normalized; 0 if absent)
 * - 3 features: Inter-hand relative wrist offset (dx, dy, dz) / scale1 (0 if absent)
 * - 2 features: Hand presence flags [hand1_present (1.0), hand2_present (1.0 or 0.0)]
 * Total: 131 features. Supports both 1-handed and 2-handed gestures seamlessly.
 */
export const FEATURE_VECTOR_SIZE = 131;

/**
 * Normalizes 21 3D landmarks for a single hand:
 * 1. Subtract wrist coordinates (landmark 0) -> translation invariance.
 * 2. Divide by max Euclidean distance from wrist -> scale invariance.
 */
function normalizeSingleHand(pts: Point3D[]): { features: number[]; scale: number } {
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

  return { features, scale };
}

/**
 * Sorts detected hands consistently:
 * If both Left and Right are present, Left comes first, Right second.
 * Otherwise, the leftmost hand on screen comes first.
 */
export function sortDetectedHands(hands: DetectedHand[]): DetectedHand[] {
  if (hands.length <= 1) return hands;
  const left = hands.find((h) => h.handedness === "Left");
  const right = hands.find((h) => h.handedness === "Right");
  if (left && right) {
    return [left, right];
  }
  return [...hands].sort(
    (a, b) => (a.landmarks[0]?.x ?? 0) - (b.landmarks[0]?.x ?? 0)
  );
}

/**
 * Extracts a 131-dimensional translation- and scale-invariant feature vector
 * that supports BOTH single-handed and two-handed poses:
 *
 * - Features 0..62: Hand 1 (21 points * 3 = 63, wrist-centered & normalized)
 * - Features 63..125: Hand 2 (21 points * 3 = 63, wrist-centered & normalized; 0 if absent)
 * - Features 126..128: Inter-hand relative wrist offset (dx, dy, dz) / scale1 (0 if absent)
 * - Features 129..130: Hand presence flags [hand1_present (1.0), hand2_present (1.0 or 0.0)]
 */
export function extractFeatureVector(hands: DetectedHand[]): number[] | null {
  if (!hands || hands.length === 0) return null;

  const validHands = hands.filter((h) => h.landmarks && h.landmarks.length >= 21);
  if (validHands.length === 0) return null;

  const sorted = sortDetectedHands(validHands);
  const hand1 = sorted[0];
  const hand2 = sorted.length > 1 ? sorted[1] : null;

  const norm1 = normalizeSingleHand(hand1.landmarks);
  const out = new Array<number>(FEATURE_VECTOR_SIZE).fill(0);

  // Hand 1 features (0..62)
  for (let i = 0; i < 63; i++) {
    out[i] = norm1.features[i];
  }

  // Hand 1 presence flag
  out[129] = 1.0;

  if (hand2) {
    const norm2 = normalizeSingleHand(hand2.landmarks);
    // Hand 2 features (63..125)
    for (let i = 0; i < 63; i++) {
      out[63 + i] = norm2.features[i];
    }

    // Relative inter-hand wrist offset (126..128), normalized by Hand 1's scale
    const w1 = hand1.landmarks[0];
    const w2 = hand2.landmarks[0];
    const refScale = norm1.scale > 1e-6 ? norm1.scale : 1;

    out[126] = (w2.x - w1.x) / refScale;
    out[127] = (w2.y - w1.y) / refScale;
    out[128] = (w2.z - w1.z) / refScale;

    // Hand 2 presence flag
    out[130] = 1.0;
  }

  return out;
}

/**
 * Creates a slightly perturbed copy of a 131-D feature vector (jitter + minor 2D rotation)
 * to improve classifier generalization on small user-collected datasets.
 */
export function augmentFeatureVector(
  features: number[],
  noiseStd = 0.018,
  maxAngleRad = 0.14
): number[] {
  const out = [...features];

  const augmentHandBlock = (offset: number) => {
    const angle = (Math.random() * 2 - 1) * maxAngleRad;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    for (let i = 0; i < 63; i += 3) {
      const x = features[offset + i];
      const y = features[offset + i + 1];
      const z = features[offset + i + 2];

      out[offset + i] = x * cosA - y * sinA + gaussianRandom() * noiseStd;
      out[offset + i + 1] = x * sinA + y * cosA + gaussianRandom() * noiseStd;
      out[offset + i + 2] = z + gaussianRandom() * noiseStd;
    }
  };

  // Augment Hand 1 (indices 0..62)
  augmentHandBlock(0);

  // If Hand 2 is present (flag at index 130 is 1.0)
  if (features[130] > 0.5) {
    augmentHandBlock(63);

    // Add slight jitter to relative inter-hand offset (indices 126..128)
    out[126] += gaussianRandom() * (noiseStd * 1.5);
    out[127] += gaussianRandom() * (noiseStd * 1.5);
    out[128] += gaussianRandom() * (noiseStd * 1.5);
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
