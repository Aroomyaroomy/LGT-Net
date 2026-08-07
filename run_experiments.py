#!/usr/bin/env python
"""
Batch experiment runner — runs the full DINO→SAM→SAM3D→Inpaint→LGT pipeline
for each of the 5 S3 panorama images with room-specific detection prompts.

Results are saved under experiments/<room_name>/ with per-stage outputs and
a summary manifest experiments/summary.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional

# Fix UnicodeEncodeError on Windows cp1252 terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure the repo root is on sys.path so imports work
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_pipeline import load_pipeline_config, run_pipeline_from_config

# ── S3 panorama URLs ──────────────────────────────────────────────
S3_BASE = (
    "https://this-is-a-valid-bucket-name-651697298829-ap-northeast-1-an"
    ".s3.ap-northeast-1.amazonaws.com"
)

# ── Per-room configuration ────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "kitchen",
        "image_url": f"{S3_BASE}/kitchen.jpg",
        "text_prompt": "cabinet. counter. stove. sink. table. chair. refrigerator.",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "min_score": None,
        "crop_size": 512,
    },
    {
        "name": "bathroom",
        "image_url": f"{S3_BASE}/bathroom.jpg",
        "text_prompt": "toilet. sink. bathtub. shower. mirror. cabinet.",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "min_score": None,
        "crop_size": 512,
    },
    {
        "name": "dim_room",
        "image_url": f"{S3_BASE}/dim_room.jpg",
        "text_prompt": "sofa. chair. table. cabinet. lamp. bed.",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "min_score": None,
        "crop_size": 512,
    },
    {
        "name": "small_square_room",
        "image_url": f"{S3_BASE}/small_square_room.jpg",
        "text_prompt": "sofa. chair. table. cabinet. bed. shelf.",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "min_score": None,
        "crop_size": 512,
    },
    {
        "name": "office",
        "image_url": f"{S3_BASE}/office.jpg",
        "text_prompt": "desk. chair. monitor. computer. cabinet. bookshelf.",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "min_score": None,
        "crop_size": 512,
    },
]

OUTPUT_ROOT = os.path.join(REPO_ROOT, "experiments")


def build_config(base_config: dict, exp: dict) -> dict:
    """Deep-copy the base config and overlay per-experiment overrides."""
    cfg = json.loads(json.dumps(base_config))  # cheap deep copy
    cfg["image_url"] = exp["image_url"]
    cfg["root_dir"] = os.path.join(OUTPUT_ROOT, exp["name"])

    # Use local pre-downloaded image for DINO/Inpaint stages (fast)
    # while keeping S3 URL for SAM/SAM3D external APIs
    local_img = os.path.join(OUTPUT_ROOT, "downloads", f"{exp['name']}.jpg")
    if os.path.isfile(local_img):
        cfg["local_image"] = os.path.abspath(local_img)
    else:
        cfg["local_image"] = exp["image_url"]  # fallback to S3

    # DINO overrides
    cfg["dino"]["text_prompt"] = exp["text_prompt"]
    cfg["dino"]["box_threshold"] = exp["box_threshold"]
    cfg["dino"]["text_threshold"] = exp["text_threshold"]

    # SAM overrides
    if exp.get("min_score") is not None:
        cfg["sam"]["min_score"] = exp["min_score"]

    # Inpaint overrides
    cfg["inpaint"]["crop_size"] = exp["crop_size"]

    # Disable interactive visualization (headless run)
    cfg["visualize"]["show"] = False

    return cfg


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    base_config_path = os.path.join(REPO_ROOT, "pipeline_config.json")
    base_config = load_pipeline_config(base_config_path)

    summary = {
        "started_at": datetime.now().isoformat(),
        "base_config": base_config_path,
        "results": [],
    }

    for i, exp in enumerate(EXPERIMENTS, 1):
        name = exp["name"]
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT {i}/{len(EXPERIMENTS)}: {name}")
        print(f"{'='*70}")
        print(f"  Image:    {exp['image_url']}")
        print(f"  Prompt:   {exp['text_prompt']}")
        print(f"  Box thr:  {exp['box_threshold']}")
        print(f"  Text thr: {exp['text_threshold']}")
        print(f"  Crop:     {exp['crop_size']}")
        print(f"{'='*70}\n")

        cfg = build_config(base_config, exp)
        t_start = time.time()

        result_entry = {
            "name": name,
            "image_url": exp["image_url"],
            "text_prompt": exp["text_prompt"],
            "box_threshold": exp["box_threshold"],
            "text_threshold": exp["text_threshold"],
            "status": "error",
            "lgt_mesh_path": None,
            "dino_job_id": None,
            "num_masks": 0,
            "num_objects_3d": 0,
            "duration_s": 0.0,
            "error": None,
        }

        try:
            pipeline_result = run_pipeline_from_config(cfg)

            result_entry["status"] = "ok"
            result_entry["dino_job_id"] = (
                pipeline_result.get("dino", {}).get("job_id")
            )
            result_entry["num_masks"] = len(
                pipeline_result.get("sam", {})
                .get("data", {})
                .get("masks", [])
            )
            result_entry["num_objects_3d"] = len(
                pipeline_result.get("sam3d", {})
                .get("data", {})
                .get("individual_glbs", [])
            )

            lgt = pipeline_result.get("lgt_net", {})
            result_entry["lgt_mesh_path"] = lgt.get("mesh_path")
            result_entry["lgt_placements_path"] = lgt.get("placements_path")
            result_entry["lgt_job_id"] = lgt.get("job_id")

            print(f"\n  ✓ {name} PASSED")
            if result_entry["lgt_mesh_path"]:
                print(f"    LGT mesh: {result_entry['lgt_mesh_path']}")

        except Exception as exc:
            result_entry["error"] = traceback.format_exc()
            print(f"\n  [FAIL] {name}: {exc}")

        result_entry["duration_s"] = round(time.time() - t_start, 1)
        summary["results"].append(result_entry)

    # ── Write summary manifest ────────────────────────────────────
    summary["finished_at"] = datetime.now().isoformat()
    summary_path = os.path.join(OUTPUT_ROOT, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print final table ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Room':<22} {'Status':<8} {'Masks':>6} {'3D Obj':>7} {'LGT Mesh':>40}")
    print(f"{'-'*22} {'-'*8} {'-'*6} {'-'*7} {'-'*40}")
    for r in summary["results"]:
        mesh = os.path.basename(r["lgt_mesh_path"] or "") or "—"
        print(
            f"{r['name']:<22} {r['status']:<8} "
            f"{r['num_masks']:>6} {r['num_objects_3d']:>7} "
            f"{mesh[:40]:>40}"
        )
    print(f"\n  Full summary: {os.path.abspath(summary_path)}")

    # ── List all LGT meshes ───────────────────────────────────────
    print(f"\n  LGT-Net Mesh Outputs:")
    for r in summary["results"]:
        if r["lgt_mesh_path"]:
            print(f"    {r['name']}: {r['lgt_mesh_path']}")

    return summary


if __name__ == "__main__":
    main()
