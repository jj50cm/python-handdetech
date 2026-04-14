"""
Fine-tune YOLOv8x on a phone detection dataset.

Two dataset options:
  Option A (Roboflow) — automatic download, requires free account + API key
  Option B (manual)  — place your own dataset in the folder structure below

Usage:
  python train.py --dataset roboflow --api-key YOUR_KEY
  python train.py --dataset manual   --data-dir datasets/phone
  python train.py --resume           # continue interrupted training

After training, the best weights are saved to:
  runs/detect/phone_finetune/weights/best.pt

Then run the detector with your fine-tuned model:
  python detect.py --source http://192.168.0.161:8080/video --model runs/detect/phone_finetune/weights/best.pt
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


# ── Hardware settings optimised for RTX 4060 8GB ─────────────────────────────
TRAIN_CFG = dict(
    epochs=100,
    imgsz=640,
    batch=8,            # safe for 8GB VRAM with amp=True; raise to 16 if comfortable
    device=0,           # GPU 0
    optimizer="AdamW",
    lr0=0.001,          # low LR for fine-tuning
    lrf=0.01,           # final LR = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    freeze=10,          # freeze first 10 backbone layers — faster, less VRAM
    amp=True,           # FP16: halves VRAM usage, ~1.5x faster
    patience=30,        # early stop if no improvement for 30 epochs
    workers=4,          # Windows: keep ≤4 to avoid dataloader overhead
    project="runs",
    name="phone_finetune",
    exist_ok=True,
    verbose=True,
    # ── Augmentation for tilted/angled phones ────────────────────────────────
    degrees=45.0,       # rotate up to ±45° — phones lying flat, tilted on table
    shear=10.0,         # shear up to ±10° — simulate camera angle shift
    perspective=0.0005, # subtle perspective warp — camera not perfectly frontal
    flipud=0.5,         # 50% flip vertical — phone upside-down on table
    # ─────────────────────────────────────────────────────────────────────────
)
# ─────────────────────────────────────────────────────────────────────────────


def download_roboflow(api_key: str, dest: str) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow (skipping opencv-headless to avoid conflict)...")
        # --no-deps avoids replacing opencv-python with opencv-python-headless
        ret = os.system(f"{sys.executable} -m pip install roboflow --no-deps")
        # install the remaining roboflow deps that aren't already present
        os.system(f"{sys.executable} -m pip install "
                  f"requests python-dotenv tqdm PyYAML requests-toolbelt "
                  f"filetype typer pillow idna")
        if ret != 0:
            print("ERROR: roboflow install failed. Try manually:")
            print(f"  {sys.executable} -m pip install roboflow --no-deps")
            sys.exit(1)
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    # RealTime Mobile Phone Usage Detection dataset (1,674 images, YOLOv8 format)
    project = rf.workspace("realtime-mobile-phone-usage-detection-in-everyday-scenarios-using-yolo") \
                .project("mobile-phone-detection-mtsje-xhoma")
    version = project.version(1)
    dataset = version.download("yolov8", location=dest)
    return Path(dataset.location) / "data.yaml"


def build_manual_yaml(data_dir: str) -> Path:
    data_dir = Path(data_dir).resolve()
    yaml_path = data_dir / "data.yaml"

    if yaml_path.exists():
        print(f"Using existing {yaml_path}")
        return yaml_path

    # Write a default yaml — user can edit class names if needed
    yaml_content = f"""path: {data_dir.as_posix()}
train: images/train
val: images/val

nc: 1
names: ['phone']
"""
    yaml_path.write_text(yaml_content)
    print(f"Created {yaml_path} — edit if your class names differ.")
    return yaml_path


def validate_manual_dataset(data_dir: str):
    base = Path(data_dir)

    # Support both standard and Roboflow-style layouts
    if (base / "train" / "images").exists():
        # Roboflow layout: train/images/, valid/images/
        train_imgs = base / "train" / "images"
        val_imgs   = base / "valid" / "images"
    else:
        # Standard layout: images/train/, images/val/
        train_imgs = base / "images" / "train"
        val_imgs   = base / "images" / "val"

    missing = [p for p in [train_imgs, val_imgs] if not p.exists()]
    if missing:
        print("\nERROR: Missing required directories:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    n_train = len(list(train_imgs.glob("*.*")))
    n_val   = len(list(val_imgs.glob("*.*")))
    print(f"Dataset: {n_train} train images, {n_val} val images")


def train(data_yaml: Path, start_weights: str = "yolov8x.pt"):
    print(f"\nStarting fine-tune from {start_weights}")
    print(f"Dataset: {data_yaml}")
    print(f"Output:  runs/detect/phone_finetune/weights/best.pt\n")

    model = YOLO(start_weights)
    model.train(data=str(data_yaml), **TRAIN_CFG)

    best = Path("runs/detect/phone_finetune/weights/best.pt")
    if best.exists():
        print(f"\nDone. Best model saved to: {best}")
        print(f"\nRun detector with fine-tuned model:")
        print(f"  python detect.py --source http://192.168.0.161:8080/video --model {best}")
    else:
        print("\nTraining finished but best.pt not found — check runs/detect/phone_finetune/")


def resume():
    # Search for last.pt under the runs directory in case the path is nested
    candidates = list(Path("runs").rglob("phone_finetune/weights/last.pt"))
    if not candidates:
        print("No interrupted training found (last.pt not found under runs/)")
        sys.exit(1)
    last = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Resuming from {last}")
    model = YOLO(str(last))
    model.train(resume=True)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8x for phone detection")
    parser.add_argument("--dataset", choices=["roboflow", "manual"], default="roboflow",
                        help="Dataset source (default: roboflow)")
    parser.add_argument("--api-key", default="",
                        help="Roboflow API key (required for --dataset roboflow)")
    parser.add_argument("--data-dir", default="datasets/phone",
                        help="Local dataset directory (for --dataset manual)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted training")
    parser.add_argument("--weights", default=None,
                        help="Starting weights (default: yolov8x.pt). Use to continue from a previous best.pt")
    args = parser.parse_args()

    if args.resume:
        resume()
        return

    if args.dataset == "roboflow":
        if not args.api_key:
            print("ERROR: --api-key is required for Roboflow download.")
            print("  Get a free key at https://roboflow.com  (Settings > API Keys)")
            sys.exit(1)
        data_yaml = download_roboflow(args.api_key, "datasets/phone")
    else:
        validate_manual_dataset(args.data_dir)
        data_yaml = build_manual_yaml(args.data_dir)

    train(data_yaml, start_weights=args.weights or "yolov8x.pt")


if __name__ == "__main__":
    main()
