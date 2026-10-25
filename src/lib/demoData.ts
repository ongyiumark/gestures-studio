import {
  CapturedSample,
  DetectedHand,
  Point3D,
  PoseClassMeta,
} from "@/types/handpose";
import { drawHandsOnCanvas, getClassColor } from "@/lib/landmarks";

/**
 * Canonical 21-landmark normalized templates for 4 classic hand gestures
 * so users can test training, live charts, and exports with a single click
 * even before turning on their webcam.
 */
const POSE_TEMPLATES: Record<string, Point3D[]> = {
  "Open Palm": [
    { x: 0.5, y: 0.78, z: 0.0 }, // 0: wrist
    { x: 0.39, y: 0.72, z: -0.02 }, // 1: thumb_cmc
    { x: 0.32, y: 0.63, z: -0.03 }, // 2: thumb_mcp
    { x: 0.27, y: 0.54, z: -0.03 }, // 3: thumb_ip
    { x: 0.22, y: 0.46, z: -0.04 }, // 4: thumb_tip
    { x: 0.42, y: 0.52, z: -0.01 }, // 5: index_mcp
    { x: 0.4, y: 0.39, z: -0.02 }, // 6: index_pip
    { x: 0.39, y: 0.29, z: -0.02 }, // 7: index_dip
    { x: 0.38, y: 0.2, z: -0.03 }, // 8: index_tip
    { x: 0.49, y: 0.5, z: -0.01 }, // 9: middle_mcp
    { x: 0.49, y: 0.36, z: -0.02 }, // 10: middle_pip
    { x: 0.49, y: 0.25, z: -0.02 }, // 11: middle_dip
    { x: 0.49, y: 0.16, z: -0.03 }, // 12: middle_tip
    { x: 0.56, y: 0.52, z: -0.01 }, // 13: ring_mcp
    { x: 0.58, y: 0.39, z: -0.02 }, // 14: ring_pip
    { x: 0.59, y: 0.29, z: -0.02 }, // 15: ring_dip
    { x: 0.6, y: 0.21, z: -0.03 }, // 16: ring_tip
    { x: 0.63, y: 0.56, z: -0.01 }, // 17: pinky_mcp
    { x: 0.66, y: 0.46, z: -0.02 }, // 18: pinky_pip
    { x: 0.68, y: 0.38, z: -0.02 }, // 19: pinky_dip
    { x: 0.7, y: 0.3, z: -0.03 }, // 20: pinky_tip
  ],
  Fist: [
    { x: 0.5, y: 0.75, z: 0.0 },
    { x: 0.41, y: 0.68, z: -0.02 },
    { x: 0.38, y: 0.58, z: -0.03 },
    { x: 0.43, y: 0.53, z: -0.04 },
    { x: 0.49, y: 0.54, z: -0.04 },
    { x: 0.43, y: 0.52, z: -0.01 },
    { x: 0.43, y: 0.48, z: -0.04 },
    { x: 0.44, y: 0.56, z: -0.05 },
    { x: 0.45, y: 0.61, z: -0.04 },
    { x: 0.49, y: 0.51, z: -0.01 },
    { x: 0.49, y: 0.47, z: -0.04 },
    { x: 0.5, y: 0.56, z: -0.05 },
    { x: 0.5, y: 0.62, z: -0.04 },
    { x: 0.55, y: 0.53, z: -0.01 },
    { x: 0.55, y: 0.49, z: -0.04 },
    { x: 0.55, y: 0.57, z: -0.05 },
    { x: 0.55, y: 0.63, z: -0.04 },
    { x: 0.61, y: 0.56, z: -0.01 },
    { x: 0.61, y: 0.52, z: -0.04 },
    { x: 0.6, y: 0.59, z: -0.05 },
    { x: 0.59, y: 0.64, z: -0.04 },
  ],
  Peace: [
    { x: 0.5, y: 0.78, z: 0.0 },
    { x: 0.41, y: 0.7, z: -0.02 },
    { x: 0.38, y: 0.61, z: -0.03 },
    { x: 0.44, y: 0.55, z: -0.04 },
    { x: 0.51, y: 0.56, z: -0.04 },
    { x: 0.44, y: 0.52, z: -0.01 },
    { x: 0.41, y: 0.38, z: -0.02 },
    { x: 0.39, y: 0.27, z: -0.03 },
    { x: 0.37, y: 0.18, z: -0.03 },
    { x: 0.51, y: 0.51, z: -0.01 },
    { x: 0.53, y: 0.37, z: -0.02 },
    { x: 0.55, y: 0.26, z: -0.03 },
    { x: 0.57, y: 0.17, z: -0.03 },
    { x: 0.56, y: 0.55, z: -0.01 },
    { x: 0.56, y: 0.52, z: -0.04 },
    { x: 0.55, y: 0.59, z: -0.05 },
    { x: 0.54, y: 0.64, z: -0.04 },
    { x: 0.62, y: 0.58, z: -0.01 },
    { x: 0.61, y: 0.55, z: -0.04 },
    { x: 0.6, y: 0.61, z: -0.05 },
    { x: 0.59, y: 0.66, z: -0.04 },
  ],
  "Thumbs Up": [
    { x: 0.48, y: 0.68, z: 0.0 },
    { x: 0.46, y: 0.56, z: -0.02 },
    { x: 0.45, y: 0.44, z: -0.03 },
    { x: 0.45, y: 0.33, z: -0.03 },
    { x: 0.45, y: 0.23, z: -0.04 },
    { x: 0.56, y: 0.5, z: -0.01 },
    { x: 0.61, y: 0.5, z: -0.04 },
    { x: 0.57, y: 0.52, z: -0.05 },
    { x: 0.53, y: 0.52, z: -0.04 },
    { x: 0.56, y: 0.56, z: -0.01 },
    { x: 0.61, y: 0.56, z: -0.04 },
    { x: 0.57, y: 0.58, z: -0.05 },
    { x: 0.53, y: 0.58, z: -0.04 },
    { x: 0.56, y: 0.62, z: -0.01 },
    { x: 0.61, y: 0.62, z: -0.04 },
    { x: 0.57, y: 0.64, z: -0.05 },
    { x: 0.53, y: 0.64, z: -0.04 },
    { x: 0.55, y: 0.68, z: -0.01 },
    { x: 0.6, y: 0.68, z: -0.04 },
    { x: 0.56, y: 0.7, z: -0.05 },
    { x: 0.52, y: 0.7, z: -0.04 },
  ],
};

function renderSyntheticThumbnail(
  hand: DetectedHand,
  className: string,
  accentColor: string
): string {
  if (typeof document === "undefined") return "";
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 240;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const grad = ctx.createLinearGradient(0, 0, 320, 240);
  grad.addColorStop(0, "#0f172a");
  grad.addColorStop(1, "#1e293b");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 320, 240);

  // Subtle grid lines
  ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
  ctx.lineWidth = 1;
  for (let x = 40; x < 320; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 240);
    ctx.stroke();
  }
  for (let y = 40; y < 240; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(320, y);
    ctx.stroke();
  }

  drawHandsOnCanvas([hand], ctx, 320, 240, {
    compact: false,
    showLabels: false,
    accentColor,
  });

  ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
  ctx.fillRect(8, 8, 120, 22);
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "600 11px Inter, system-ui, sans-serif";
  ctx.fillText(`Demo: ${className}`, 14, 23);

  return canvas.toDataURL("image/jpeg", 0.85);
}

export function generateDemoDataset(samplesPerClass = 8): {
  classes: PoseClassMeta[];
  samples: CapturedSample[];
} {
  const classNames = Object.keys(POSE_TEMPLATES);
  const now = Date.now();

  const classes: PoseClassMeta[] = classNames.map((name, idx) => ({
    name,
    color: getClassColor(idx),
    createdAt: now + idx,
  }));

  const samples: CapturedSample[] = [];

  classNames.forEach((className, classIdx) => {
    const basePoints = POSE_TEMPLATES[className];
    const color = getClassColor(classIdx);

    for (let i = 0; i < samplesPerClass; i++) {
      const offsetX = (Math.random() - 0.5) * 0.08;
      const offsetY = (Math.random() - 0.5) * 0.08;
      const scale = 0.92 + Math.random() * 0.16;

      const landmarks: Point3D[] = basePoints.map((pt) => ({
        x: Math.min(
          0.95,
          Math.max(
            0.05,
            0.5 + (pt.x - 0.5) * scale + offsetX + (Math.random() - 0.5) * 0.015
          )
        ),
        y: Math.min(
          0.95,
          Math.max(
            0.05,
            0.5 + (pt.y - 0.5) * scale + offsetY + (Math.random() - 0.5) * 0.015
          )
        ),
        z: pt.z * scale + (Math.random() - 0.5) * 0.01,
      }));

      const hand: DetectedHand = {
        handedness: i % 2 === 0 ? "Right" : "Left",
        score: Number((0.92 + Math.random() * 0.07).toFixed(2)),
        landmarks,
        worldLandmarks: landmarks.map((p) => ({
          x: (p.x - 0.5) * 0.2,
          y: (p.y - 0.5) * 0.2,
          z: p.z * 0.2,
        })),
      };

      const imageDataUrl = renderSyntheticThumbnail(hand, className, color);

      samples.push({
        id: `demo_${classIdx}_${i}_${Math.random().toString(36).slice(2, 7)}`,
        className,
        timestamp: now + i * 100,
        imageDataUrl,
        width: 320,
        height: 240,
        hands: [hand],
      });
    }
  });

  return { classes, samples };
}
