import argparse
import cv2
import socket
import threading
import time
import torch
from ultralytics import YOLO

# COCO class index for "cell phone" (base model).
# Fine-tuned model has only 1 class so this becomes 0 — set via --model flag automatically.
PHONE_CLASS_ID = 67
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

MODES = {
    "gpu": dict(model_path="yolov8x.pt", device="cuda:0", imgsz=1280, conf=0.15, half=True,  label="GPU  (YOLOv8x CUDA FP16)"),
    "cpu": dict(model_path="yolov8n.pt", device="cpu",    imgsz=320,  conf=0.4,  label="CPU  (YOLOv8n PyTorch)"),
}


def open_camera(source):
    # Convert numeric string to int so cv2 uses a device index, not a filename
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # keep as string URL

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source!r}")
    return cap


class LatestFrameReader:
    """Background thread that continuously grabs frames, keeping only the latest.
    This prevents the buffer from filling up during slow inference, eliminating lag."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            with self._lock:
                self.ok = ok
                self.frame = frame

    def read(self):
        with self._lock:
            return self.ok, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


class PingMonitor:
    """Measures TCP ping to host:port every second in a background thread."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.ping_ms = -1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                start = time.perf_counter()
                with socket.create_connection((self.host, self.port), timeout=2):
                    pass
                self.ping_ms = (time.perf_counter() - start) * 1000
            except Exception:
                self.ping_ms = -1
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def load_model(mode_key, custom_path=None):
    cfg = MODES[mode_key]
    path = custom_path if (custom_path and mode_key == "gpu") else cfg["model_path"]
    print(f"Loading {cfg['label']} from {path}...")
    model = YOLO(path)
    print("Ready.")
    return model


def run_inference(model, frame, cfg, phone_class_id):
    # fine-tuned model (phone_class_id=None): no class filter, detect everything
    classes = None if phone_class_id is None else [phone_class_id]
    return model(
        frame,
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        iou=0.45,
        device=cfg["device"],
        classes=classes,
        verbose=False,
    )


def draw_boxes(frame, results, phone_class_id):
    for result in results:
        for box in result.boxes:
            if phone_class_id is not None and int(box.cls[0]) != phone_class_id:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            label = f"Phone {conf:.0%}"
            label_y = y1 - 8 if y1 > 20 else y1 + 18
            cv2.putText(frame, label, (x1, label_y), FONT, 0.6, TEXT_COLOR, 2)


def main():
    parser = argparse.ArgumentParser(description="Hand-held phone detector")
    parser.add_argument(
        "--source",
        default="http://localhost:8080/video",
        help="Camera source: device index (0, 1, …) or stream URL.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to fine-tuned model weights (e.g. runs/detect/phone_finetune/weights/best.pt). "
             "If omitted, uses the default yolov8x.pt.",
    )
    args = parser.parse_args()

    # Fine-tuned model: don't filter by class (model already specialized for phones)
    # Base COCO model: filter to class 67 (cell phone)
    phone_class_id = None if args.model else PHONE_CLASS_ID
    if args.model:
        print(f"Using fine-tuned model: {args.model}  (no class filter)")

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU mode.")
        initial_mode = "cpu"
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        initial_mode = "gpu"

    mode_key = initial_mode
    models = {initial_mode: load_model(initial_mode, custom_path=args.model)}
    other = "cpu" if initial_mode == "gpu" else "gpu"
    models[other] = None

    print(f"Opening camera source: {args.source!r}")
    cap = open_camera(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    reader = LatestFrameReader(cap)

    # Start ping monitor if source is a URL
    ping_monitor = None
    if isinstance(args.source, str) and args.source.startswith("http"):
        from urllib.parse import urlparse
        parsed = urlparse(args.source)
        ping_monitor = PingMonitor(parsed.hostname, parsed.port or 80)

    cv2.namedWindow("Phone Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Phone Detector", 640, 480)

    prev_time = time.time()
    print("Controls: M = toggle GPU/CPU mode | Q = quit")

    while True:
        ret, frame = reader.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        cfg = MODES[mode_key]
        results = run_inference(models[mode_key], frame, cfg, phone_class_id)
        draw_boxes(frame, results, phone_class_id)

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), FONT, 0.8, (0, 200, 255), 2)
        if ping_monitor is not None:
            p = ping_monitor.ping_ms
            ping_text = f"Ping: {p:.0f} ms" if p >= 0 else "Ping: --"
            ping_color = (0, 255, 0) if p < 50 else (0, 165, 255) if p < 150 else (0, 0, 255)
            cv2.putText(frame, ping_text, (10, 58), FONT, 0.8, ping_color, 2)
        mode_color = (0, 200, 255) if mode_key == "gpu" else (200, 200, 0)
        cv2.putText(frame, cfg["label"], (10, 88), FONT, 0.6, mode_color, 2)
        cv2.putText(frame, "M: switch mode | Q: quit", (10, frame.shape[0] - 10), FONT, 0.5, (180, 180, 180), 1)

        cv2.imshow("Phone Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            mode_key = "cpu" if mode_key == "gpu" else "gpu"
            if models[mode_key] is None:
                print(f"Loading {MODES[mode_key]['label']} for the first time...")
                models[mode_key] = load_model(mode_key)
            print(f"Switched to: {MODES[mode_key]['label']}")

    reader.stop()
    if ping_monitor is not None:
        ping_monitor.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
