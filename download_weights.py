#!/usr/bin/env python3
"""
Download model weights to the host filesystem (one-time setup).

Weights are mounted as Docker volumes at runtime — NOT baked into images, NOT stored in Git.
This matches the aroomy-worker convention.

Usage:
    python download_weights.py              # download all weights
    python download_weights.py --dino-only  # GroundingDINO only
    python download_weights.py --lgt-only   # LGT-Net only
    python download_weights.py --force      # re-download even if exists

Output:
    ./GroundingDINO/weights/groundingdino_swint_ogc.pth   (662 MB)
    ./checkpoints/SWG_Transformer_LGT_Net/zind/best.pkl   (456 MB)
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# GroundingDINO — from HuggingFace
# ---------------------------------------------------------------------------
DINO_WEIGHT_URL = (
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
    "groundingdino_swint_ogc.pth"
)
DINO_WEIGHT_DIR = ROOT / "GroundingDINO" / "weights"
DINO_WEIGHT_PATH = DINO_WEIGHT_DIR / "groundingdino_swint_ogc.pth"

# ---------------------------------------------------------------------------
# LGT-Net Zind checkpoint — from Google Drive
# ---------------------------------------------------------------------------
LGT_CKPT_GDRIVE_ID = "1PzBj-dfDfH_vevgSkRe5kczW0GVl_43I"
LGT_CKPT_DIR = ROOT / "checkpoints" / "SWG_Transformer_LGT_Net" / "zind"
LGT_CKPT_PATH = LGT_CKPT_DIR / "best.pkl"


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def download_dino(force: bool = False) -> bool:
    """Download GroundingDINO weights from HuggingFace."""
    DINO_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    if DINO_WEIGHT_PATH.exists() and not force:
        print(f"[SKIP] GroundingDINO weights already exist ({_file_size_mb(DINO_WEIGHT_PATH):.0f} MB)")
        print(f"       {DINO_WEIGHT_PATH}")
        return True

    print(f"[DOWNLOAD] GroundingDINO weights (662 MB)")
    print(f"           from: {DINO_WEIGHT_URL}")
    print(f"           to:   {DINO_WEIGHT_PATH}")

    try:
        import urllib.request
        import shutil

        with urllib.request.urlopen(DINO_WEIGHT_URL) as resp, open(DINO_WEIGHT_PATH, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        print(f"        Manual download: {DINO_WEIGHT_URL}")
        print(f"        Save as:         {DINO_WEIGHT_PATH}")
        return False

    print(f"[OK] GroundingDINO weights ready ({_file_size_mb(DINO_WEIGHT_PATH):.0f} MB)")
    return True


def download_lgt(force: bool = False) -> bool:
    """Download LGT-Net Zind checkpoint from Google Drive."""
    LGT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if LGT_CKPT_PATH.exists() and not force:
        print(f"[SKIP] LGT-Net Zind checkpoint already exists ({_file_size_mb(LGT_CKPT_PATH):.0f} MB)")
        print(f"       {LGT_CKPT_PATH}")
        return True

    print(f"[DOWNLOAD] LGT-Net Zind checkpoint (456 MB)")
    print(f"           from: Google Drive (id={LGT_CKPT_GDRIVE_ID})")
    print(f"           to:   {LGT_CKPT_PATH}")

    try:
        import gdown
        gdown.download(
            f"https://drive.google.com/uc?id={LGT_CKPT_GDRIVE_ID}",
            str(LGT_CKPT_PATH),
            quiet=False,
        )
    except ImportError:
        print("[ERROR] gdown is not installed. Run: pip install gdown", file=sys.stderr)
        print(f"        Then re-run: python {__file__} --lgt-only", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        print(f"        Manual: https://drive.google.com/uc?id={LGT_CKPT_GDRIVE_ID}")
        print(f"        Save as: {LGT_CKPT_PATH}")
        return False

    if LGT_CKPT_PATH.exists():
        print(f"[OK] LGT-Net checkpoint ready ({_file_size_mb(LGT_CKPT_PATH):.0f} MB)")
        return True
    else:
        print("[ERROR] Download seemed to succeed but file not found.", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download LGT pipeline model weights")
    parser.add_argument("--dino-only", action="store_true", help="Download GroundingDINO only")
    parser.add_argument("--lgt-only", action="store_true", help="Download LGT-Net only")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    do_dino = not args.lgt_only
    do_lgt = not args.dino_only

    ok = True
    if do_dino:
        ok &= download_dino(force=args.force)
    if do_lgt:
        ok &= download_lgt(force=args.force)

    if ok:
        print("\n[DONE] All weights downloaded. Run: docker compose up -d --build")
    else:
        print("\n[WARN] Some downloads failed. See instructions above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
