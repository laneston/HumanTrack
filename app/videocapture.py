import sys
from pathlib import Path

# import warnings

current_path = Path(__file__).resolve()
project_parent = (
    current_path.parent.parent
)  # Adjust the times of .parent according to the actual structure


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"destination directory does not exist: {project_parent}")

try:
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort

except ImportError as e:
    print(f"Import failed: {e}")



import cv2


# 初始化模型和追踪器
model = YOLO("yolo11n.pt")  # 请确保已下载权重文件
tracker = DeepSort(max_age=5)  # 创建DeepSort追踪器

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    """
    YOLOv10检测
    Results 对象属性​：
    **.boxes**: 包含检测框信息的子对象。
    **.boxes.xyxy**: 所有检测框的坐标，格式为 [[x1, y1, x2, y2], ...]（Tensor 类型）。
    **.boxes.conf**: 每个检测框的置信度（Tensor 类型）。
    **.boxes.cls**: 每个检测框的类别 ID（Tensor 类型）。
    """
    results = model(frame, verbose=False)

    # 准备检测结果
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            if box.cls == 0:  # 过滤person类（COCO类别0）
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0]
                detections.append(([x1, y1, x2, y2], confidence, 0))

    # 更新追踪器
    tracks = tracker.update_tracks(detections, frame=frame)

    # 绘制追踪结果
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb().astype(int)

        # 绘制边界框和ID
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # 显示结果
    cv2.imshow("Human Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
