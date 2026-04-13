import cv2
import time
import torch
from ultralytics import YOLO

# COCO class index for "cell phone"
PHONE_CLASS_ID = 67
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

MODES = {
    "gpu": dict(model_path="yolov8s.pt", device="cuda:0", imgsz=640, conf=0.3, label="GPU  (YOLOv8s CUDA)"),
    "cpu": dict(model_path="yolov8n.pt", device="cpu",    imgsz=320, conf=0.4, label="CPU  (YOLOv8n PyTorch)"),
}


def load_model(mode_key):
    cfg = MODES[mode_key]
    print(f"Loading {cfg['label']}...")
    model = YOLO(cfg["model_path"])
    print("Ready.")
    return model


def run_inference(model, frame, cfg):
    return model(
        frame,
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        iou=0.45,
        device=cfg["device"],
        classes=[PHONE_CLASS_ID],
        verbose=False,
    )


def draw_boxes(frame, results):
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != PHONE_CLASS_ID:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            label = f"Phone {conf:.0%}"
            label_y = y1 - 8 if y1 > 20 else y1 + 18
            cv2.putText(frame, label, (x1, label_y), FONT, 0.6, TEXT_COLOR, 2)


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU mode.")
        initial_mode = "cpu"
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        initial_mode = "gpu"

    mode_key = initial_mode
    models = {initial_mode: load_model(initial_mode)}
    # Lazily load the other mode on first switch
    other = "cpu" if initial_mode == "gpu" else "gpu"
    models[other] = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):
        cap.read()

    prev_time = time.time()

    print("Controls: M = toggle GPU/CPU mode | Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame.")
            break

        cfg = MODES[mode_key]
        results = run_inference(models[mode_key], frame, cfg)
        draw_boxes(frame, results)

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), FONT, 0.8, (0, 200, 255), 2)
        mode_color = (0, 200, 255) if mode_key == "gpu" else (200, 200, 0)
        cv2.putText(frame, cfg["label"], (10, 58), FONT, 0.6, mode_color, 2)
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
