import cv2
import time
from ultralytics import YOLO

# COCO class index for "cell phone"
PHONE_CLASS_ID = 67
CONF_THRESHOLD = 0.3
MODEL_SIZE = 416

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# CoreML model path (exported once, reused every run)
COREML_MODEL = "yolov8s.mlpackage"


def export_coreml():
    import os
    if os.path.exists(COREML_MODEL):
        print(f"CoreML model already exists at {COREML_MODEL}")
        return COREML_MODEL
    print("Exporting YOLOv8s to CoreML (one-time, ~60s)...")
    base = YOLO("yolov8s.pt")  # small model: much better accuracy than nano
    path = base.export(format="coreml", imgsz=MODEL_SIZE, nms=False)  # let ultralytics handle NMS (more accurate)
    print(f"Exported to: {path}")
    return path


def main():
    model_path = export_coreml()
    print("Loading CoreML model...")
    model = YOLO(model_path, task="detect")
    print("Model loaded. Opening camera...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # macOS needs a moment for camera to initialize
    for _ in range(5):
        cap.read()

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame.")
            break

        results = model(
            frame,
            imgsz=MODEL_SIZE,
            conf=CONF_THRESHOLD,
            classes=[PHONE_CLASS_ID],
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != PHONE_CLASS_ID:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

                label = f"Phone {conf:.0%}"
                label_y = y1 - 8 if y1 > 20 else y1 + 18
                cv2.putText(frame, label, (x1, label_y), FONT, 0.6, TEXT_COLOR, 2)

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), FONT, 0.8, (0, 200, 255), 2)

        cv2.imshow("Phone Detector — press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
