#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def render(manifest_path: Path, summary_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    poses = [row["nearest_slam_pose"]["position_xyz"] for row in manifest if row.get("nearest_slam_pose")]
    if poses:
        xs = [p[0] for p in poses]
        zs = [p[2] for p in poses]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        scale_x = max(max_x - min_x, 1e-6)
        scale_z = max(max_z - min_z, 1e-6)
        points = []
        for x, z in zip(xs, zs, strict=True):
            px = 94 + (x - min_x) / scale_x * 430
            py = 232 - (z - min_z) / scale_z * 150
            points.append((px, py))
    else:
        points = [(94, 190), (220, 150), (360, 180), (524, 120)]
    path = " ".join([("M" if idx == 0 else "L") + f"{x:.1f} {y:.1f}" for idx, (x, y) in enumerate(points)])
    circles = "\n".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#e5edf8" stroke="#020617" stroke-width="3"/>' for x, y in points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360">
  <rect width="760" height="360" rx="28" fill="#020617"/>
  <text x="48" y="48" fill="#f8fafc" font-size="26" font-weight="700" font-family="Inter, Arial">Frame-to-SLAM Manifest</text>
  <text x="48" y="76" fill="#94a3b8" font-size="15" font-family="Inter, Arial">{len(manifest)} extracted frames · {summary['num_exported_poses']} exported SLAM poses · {summary['num_point_cloud_points']} point-cloud points</text>
  <rect x="62" y="104" width="560" height="170" rx="22" fill="#07101e" stroke="#263b5f"/>
  <path d="{path}" fill="none" stroke="#60a5fa" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  {circles}
  <text x="48" y="320" fill="#64748b" font-size="13" font-family="Inter, Arial">Each point is an extracted frame linked to its nearest SLAM pose.</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def render_contact_sheet(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cards = []
    for idx, row in enumerate(manifest[:12]):
        col = idx % 4
        grid_row = idx // 4
        x = 48 + col * 168
        y = 76 + grid_row * 78
        pose = row.get("nearest_slam_pose") or {}
        cards.append(f"""
  <rect x="{x}" y="{y}" width="148" height="58" rx="14" fill="#07101e" stroke="#263b5f"/>
  <text x="{x + 14}" y="{y + 23}" fill="#e5edf8" font-family="Inter,Arial" font-size="13">{row.get("image", "frame")}</text>
  <text x="{x + 14}" y="{y + 44}" fill="#9fb0c9" font-family="Inter,Arial" font-size="12">src {row.get("source_frame")} · pose {pose.get("timestamp", "n/a")}</text>
""")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360">
  <rect width="760" height="360" rx="28" fill="#020617"/>
  <text x="48" y="48" fill="#f8fafc" font-size="26" font-weight="700" font-family="Inter, Arial">Frame Contact Sheet</text>
  {''.join(cards)}
  <text x="48" y="326" fill="#64748b" font-size="13" font-family="Inter, Arial">Each card links an extracted image to the nearest SLAM pose timestamp.</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> int:
    render(Path("outputs/sample_demo/frames_manifest.json"), Path("outputs/sample_demo/slam_summary.json"), Path("docs/assets/reconstruction_manifest.svg"))
    render_contact_sheet(Path("outputs/sample_demo/frames_manifest.json"), Path("docs/assets/frame_contact_sheet.svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
