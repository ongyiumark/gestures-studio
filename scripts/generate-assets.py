import subprocess
import os

LANDMARKS_OPEN_PALM = [
    {"x": 0.50, "y": 0.78, "z": 0.0},   # 0: wrist
    {"x": 0.39, "y": 0.72, "z": -0.02},  # 1: thumb_cmc
    {"x": 0.32, "y": 0.63, "z": -0.03},  # 2: thumb_mcp
    {"x": 0.27, "y": 0.54, "z": -0.03},  # 3: thumb_ip
    {"x": 0.22, "y": 0.46, "z": -0.04},  # 4: thumb_tip
    {"x": 0.42, "y": 0.52, "z": -0.01},  # 5: index_mcp
    {"x": 0.40, "y": 0.39, "z": -0.02},  # 6: index_pip
    {"x": 0.39, "y": 0.29, "z": -0.02},  # 7: index_dip
    {"x": 0.38, "y": 0.20, "z": -0.03},  # 8: index_tip
    {"x": 0.49, "y": 0.50, "z": -0.01},  # 9: middle_mcp
    {"x": 0.49, "y": 0.36, "z": -0.02},  # 10: middle_pip
    {"x": 0.49, "y": 0.25, "z": -0.02},  # 11: middle_dip
    {"x": 0.49, "y": 0.16, "z": -0.03},  # 12: middle_tip
    {"x": 0.56, "y": 0.52, "z": -0.01},  # 13: ring_mcp
    {"x": 0.58, "y": 0.39, "z": -0.02},  # 14: ring_pip
    {"x": 0.59, "y": 0.29, "z": -0.02},  # 15: ring_dip
    {"x": 0.60, "y": 0.21, "z": -0.03},  # 16: ring_tip
    {"x": 0.63, "y": 0.56, "z": -0.01},  # 17: pinky_mcp
    {"x": 0.66, "y": 0.46, "z": -0.02},  # 18: pinky_pip
    {"x": 0.68, "y": 0.38, "z": -0.02},  # 19: pinky_dip
    {"x": 0.70, "y": 0.30, "z": -0.03},  # 20: pinky_tip
]

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

def generate_hand_elements(center_x, center_y, scale, line_width=6):
    pts = []
    for p in LANDMARKS_OPEN_PALM:
        px = center_x + (p["x"] - 0.46) * scale
        py = center_y + (p["y"] - 0.48) * scale
        pts.append((px, py))
    
    elems = []
    
    # Palm polygon
    palm_idx = [0, 1, 2, 5, 9, 13, 17]
    poly_str = " ".join([f"{pts[i][0]:.1f},{pts[i][1]:.1f}" for i in palm_idx])
    elems.append(f'<polygon points="{poly_str}" fill="#38bdf8" fill-opacity="0.10" />')

    # Connections layer 1: Outer soft glow
    for start, end in HAND_CONNECTIONS:
        x1, y1 = pts[start]
        x2, y2 = pts[end]
        elems.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#38bdf8" stroke-width="{line_width*2.8:.1f}" stroke-opacity="0.25" stroke-linecap="round"/>')

    # Connections layer 2: Primary neon line
    for start, end in HAND_CONNECTIONS:
        x1, y1 = pts[start]
        x2, y2 = pts[end]
        elems.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#38bdf8" stroke-width="{line_width:.1f}" stroke-linecap="round"/>')

    # Connections layer 3: Inner highlight core
    for start, end in HAND_CONNECTIONS:
        x1, y1 = pts[start]
        x2, y2 = pts[end]
        elems.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#e0f2fe" stroke-width="{line_width*0.35:.1f}" stroke-linecap="round"/>')

    # 21 Joints
    tips = {4, 8, 12, 16, 20}
    for idx, (x, y) in enumerate(pts):
        if idx == 0: # Wrist (landmark 0)
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*2.5:.1f}" fill="#fbbf24" fill-opacity="0.3" />')
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*1.7:.1f}" fill="#fbbf24" stroke="#ffffff" stroke-width="{line_width*0.4:.1f}" />')
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*0.7:.1f}" fill="#ffffff" />')
        elif idx in tips: # Fingertips (4, 8, 12, 16, 20)
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*2.2:.1f}" fill="#38bdf8" fill-opacity="0.35" />')
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*1.4:.1f}" fill="#ffffff" stroke="#38bdf8" stroke-width="{line_width*0.45:.1f}" />')
        else: # Knuckles and inter-finger joints
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*1.6:.1f}" fill="#ea580c" fill-opacity="0.25" />')
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*1.1:.1f}" fill="#ea580c" stroke="#38bdf8" stroke-width="{line_width*0.35:.1f}" />')
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{line_width*0.45:.1f}" fill="#fed7aa" />')

    return "\n".join(elems)

def build_icon_svg():
    width = 1024
    height = 1024
    hand_markup = generate_hand_elements(512, 530, scale=1280, line_width=13)
    
    grid = []
    for x in range(128, 1024, 128):
        grid.append(f'<line x1="{x}" y1="0" x2="{x}" y2="1024" stroke="#38bdf8" stroke-opacity="0.08" stroke-width="2"/>')
    for y in range(128, 1024, 128):
        grid.append(f'<line x1="0" y1="{y}" x2="1024" y2="{y}" stroke="#38bdf8" stroke-opacity="0.08" stroke-width="2"/>')
    grid_str = "\n".join(grid)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0a1826" />
      <stop offset="50%" stop-color="#143654" />
      <stop offset="100%" stop-color="#0e2337" />
    </linearGradient>
    <radialGradient id="centerGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.25" />
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0" />
    </radialGradient>
  </defs>

  <rect width="1024" height="1024" fill="url(#bgGrad)" />
  {grid_str}
  <circle cx="512" cy="530" r="440" fill="url(#centerGlow)" />
  <circle cx="512" cy="530" r="410" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="12 16" stroke-opacity="0.3" />
  {hand_markup}
</svg>'''
    return svg

def build_og_svg():
    width = 1200
    height = 630
    
    # Hand placed exactly inside the card (local coords: center_x=230, center_y=265)
    hand_markup = generate_hand_elements(230, 265, scale=480, line_width=6)
    
    grid = []
    for x in range(60, 1200, 60):
        grid.append(f'<line x1="{x}" y1="0" x2="{x}" y2="630" stroke="#38bdf8" stroke-opacity="0.05" stroke-width="1"/>')
    for y in range(60, 630, 60):
        grid.append(f'<line x1="0" y1="{y}" x2="1200" y2="{y}" stroke="#38bdf8" stroke-opacity="0.05" stroke-width="1"/>')
    grid_str = "\n".join(grid)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="ogBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#081420" />
      <stop offset="45%" stop-color="#0f263d" />
      <stop offset="100%" stop-color="#0b1b2b" />
    </linearGradient>
    <linearGradient id="cardGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#152e47" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#0d1e30" stop-opacity="0.98" />
    </linearGradient>
    <linearGradient id="titleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#ffffff" />
      <stop offset="60%" stop-color="#fed7aa" />
      <stop offset="100%" stop-color="#f59e0b" />
    </linearGradient>
    <radialGradient id="stageGlow" cx="230" cy="265" r="220" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.30" />
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0" />
    </radialGradient>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="url(#ogBg)"/>
  {grid_str}

  <!-- Left Side: Branding and Features -->
  <g transform="translate(80, 95)">
    <!-- Category Badge -->
    <rect x="0" y="0" width="310" height="34" rx="17" fill="#38bdf8" fill-opacity="0.12" stroke="#38bdf8" stroke-opacity="0.4" stroke-width="1.5"/>
    <circle cx="20" cy="17" r="4.5" fill="#34d399" />
    <text x="34" y="22" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="700" fill="#bae6fd" letter-spacing="1">GOOGLE MEDIAPIPE + TFJS</text>

    <!-- Main Title -->
    <text x="0" y="105" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="56" font-weight="800" fill="url(#titleGrad)" letter-spacing="-1">Hand Pose Studio</text>
    
    <!-- Subtitle -->
    <text x="0" y="150" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="22" font-weight="500" fill="#93c5fd">
      3D Hand Pose Dataset Builder &amp; In-Browser ML
    </text>

    <!-- Description -->
    <text x="0" y="210" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="16" fill="#cbd5e1">
      Collect custom gesture datasets with 21 skeletal landmarks,
    </text>
    <text x="0" y="235" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="16" fill="#cbd5e1">
      train neural networks on client-side CPU, and export models.
    </text>

    <!-- Badges Row -->
    <g transform="translate(0, 290)">
      <rect x="0" y="0" width="170" height="36" rx="10" fill="#0c1d2e" stroke="#1e3a5a" stroke-width="1.5"/>
      <text x="14" y="23" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="600" fill="#f8fafc">✓ 21 3D Landmarks</text>

      <rect x="184" y="0" width="162" height="36" rx="10" fill="#0c1d2e" stroke="#1e3a5a" stroke-width="1.5"/>
      <text x="198" y="23" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="600" fill="#f8fafc">✓ 100% In-Browser</text>

      <rect x="360" y="0" width="168" height="36" rx="10" fill="#0c1d2e" stroke="#1e3a5a" stroke-width="1.5"/>
      <text x="374" y="23" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="600" fill="#f8fafc">✓ No Backend Needed</text>
    </g>
  </g>

  <!-- Right Side: 3D Hand Tracker Stage Card -->
  <g transform="translate(680, 60)">
    <!-- Stage Card Frame -->
    <rect x="0" y="0" width="460" height="510" rx="28" fill="url(#cardGrad)" stroke="#22486d" stroke-width="2"/>
    
    <!-- Stage Header -->
    <rect x="0" y="0" width="460" height="54" rx="28" fill="#132c45" fill-opacity="0.9"/>
    <rect x="0" y="28" width="460" height="26" fill="#132c45" fill-opacity="0.9"/>
    <circle cx="30" cy="27" r="4.5" fill="#34d399"/>
    <text x="44" y="32" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#ffffff" letter-spacing="0.5">TRACKER ACTIVE</text>
    <text x="390" y="32" font-family="monospace" font-size="13" font-weight="700" fill="#fbbf24">60 FPS</text>
    <line x1="0" y1="54" x2="460" y2="54" stroke="#22486d" stroke-width="1.5"/>

    <!-- Hand Radial Glow -->
    <circle cx="230" cy="265" r="200" fill="url(#stageGlow)"/>

    <!-- The 21 Landmarks Hand Skeleton (Directly inside Card) -->
    {hand_markup}

    <!-- Floating Handedness / Confidence Badge -->
    <rect x="30" y="442" width="165" height="34" rx="8" fill="#0b1724" fill-opacity="0.9" stroke="#38bdf8" stroke-width="1.5"/>
    <text x="44" y="464" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#ffffff">Right Hand  98%</text>

    <!-- Landmark counter tag -->
    <rect x="255" y="442" width="175" height="34" rx="8" fill="#0b1724" fill-opacity="0.9" stroke="#ea580c" stroke-width="1.5"/>
    <text x="268" y="464" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#fed7aa">21 / 21 Landmarks</text>
  </g>
</svg>'''
    return svg

with open("/tmp/icon.svg", "w") as f:
    f.write(build_icon_svg())

with open("/tmp/og.svg", "w") as f:
    f.write(build_og_svg())

print("Regenerated SVGs.")
