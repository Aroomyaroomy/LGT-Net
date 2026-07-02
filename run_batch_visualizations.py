from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg"
}


def find_images(input_dir: Path) -> list[Path]:
    images = [
        path for path in input_dir.glob("*") 
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LGT-Net inference on every image in directory then save visualizations."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("src/output_batch"))
    parser.add_argument("--inference_script", type=Path, default=Path("inference.py"))
    parser.add_argument("--post_processing", choices=["manhattan", "atalanta", "original"], default="manhattan")
    parser.add_argument("--device", default="cuda",)
    parser.add_argument("--visualize_3d", action="store_true")
    parser.add_argument("--output_3d", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    cfg = args.cfg.resolve()
    output_root = args.output_root.resolve()
    inference_script = args.inference_script.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    if not cfg.exists():
        raise FileNotFoundError(f"Config file does not exist: {cfg}")

    if not inference_script.exists():
        raise FileNotFoundError(f"inference.py not found: {inference_script}")

    images = find_images(input_dir)

    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    print(f"Found {len(images)} image(s).")
    print(f"Config: {cfg}")
    print(f"Output root: {output_root}")
    print("-" * 80)

    failures: list[tuple[Path, int]] = []

    for index, image_path in enumerate(images, start=1):
        relative_stem = image_path.relative_to(input_dir).with_suffix("")
        image_output_dir = output_root / relative_stem.parent / relative_stem.name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        expected_json = image_output_dir / f"{image_path.stem}_pred.json"

        if expected_json.exists() and not args.overwrite:
            print(f"[{index}/{len(images)}] Skipping existing: {image_path.name}")
            continue

        cmd = [
            sys.executable,
            str(inference_script),
            "--cfg",
            str(cfg),
            "--img_glob",
            str(image_path),
            "--output_dir",
            str(image_output_dir),
            "--post_processing",
            args.post_processing,
            "--device",
            args.device,
        ]

        if args.visualize_3d:
            cmd.append("--visualize_3d")

        if args.output_3d:
            cmd.append("--output_3d")

        print(f"[{index}/{len(images)}] Running: {image_path}")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

        result = subprocess.run(cmd)

        if result.returncode != 0:
            failures.append((image_path, result.returncode))
            print(f"FAILED: {image_path} with exit code {result.returncode}")

            if not args.keep_going:
                raise SystemExit(result.returncode)

    print("-" * 80)

    if failures:
        print(f"Finished with {len(failures)} failure(s):")
        for image_path, return_code in failures:
            print(f"  {return_code}: {image_path}")
        raise SystemExit(1)

    print("Finished successfully.")


if __name__ == "__main__":
    main()