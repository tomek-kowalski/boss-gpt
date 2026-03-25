# -----------------------------
# Object-level diff using YOLOv8
# -----------------------------
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
root = r"E:\boss-gpt"
reports = os.path.join(root, "reports")
os.makedirs(reports, exist_ok=True)

path1 = os.path.join(root, "images", "image-1.png")
path2 = os.path.join(root, "images", "image-2.png")

csv_path = os.path.join(reports, "object_diff_report.csv")
diff_img_path = os.path.join(reports, "diff_objects.png")

# -----------------------------
# Load images
# -----------------------------
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# -----------------------------
# Load YOLOv8 model (pretrained)
# -----------------------------
model = YOLO("yolov8n.pt")  # small, fast; replace with yolov8l.pt for more accuracy

# -----------------------------
# Detect objects
# -----------------------------
results1 = model(img1_rgb)[0]
results2 = model(img2_rgb)[0]

# Extract detections: class, bbox (x1, y1, x2, y2), confidence
def extract_detections(res):
    dets = []
    for cls, box, conf in zip(res.boxes.cls.cpu().numpy(),
                              res.boxes.xyxy.cpu().numpy(),
                              res.boxes.conf.cpu().numpy()):
        dets.append({
            'class': int(cls),
            'bbox': box,       # [x1, y1, x2, y2]
            'conf': float(conf)
        })
    return dets

dets1 = extract_detections(results1)
dets2 = extract_detections(results2)

# -----------------------------
# Match objects by class + IoU
# -----------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Threshold for considering "matched"
iou_thresh = 0.3

matched1 = set()
matched2 = set()
diff_report = []

for i1, d1 in enumerate(dets1):
    best_iou = 0
    best_j = -1
    for j2, d2 in enumerate(dets2):
        if j2 in matched2:
            continue
        if d1['class'] != d2['class']:
            continue
        iou_val = iou(d1['bbox'], d2['bbox'])
        if iou_val > best_iou:
            best_iou = iou_val
            best_j = j2
    if best_iou >= iou_thresh:
        matched1.add(i1)
        matched2.add(best_j)
    else:
        # object in image1 missing in image2
        diff_report.append({
            'image': 'image_1_only',
            'class': d1['class'],
            'bbox': d1['bbox'].tolist()
        })

# Objects in image2 not matched → new objects
for j2, d2 in enumerate(dets2):
    if j2 not in matched2:
        diff_report.append({
            'image': 'image_2_only',
            'class': d2['class'],
            'bbox': d2['bbox'].tolist()
        })

# Save CSV
df_diff = pd.DataFrame(diff_report)
df_diff.to_csv(csv_path, index=False)
print(f"Saved object-level diff CSV: {csv_path}")

# -----------------------------
# Create diff visualization
# -----------------------------
img1_vis = img1_rgb.copy()
img2_vis = img2_rgb.copy()

# Color map for classes
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]

for entry in diff_report:
    bbox = [int(x) for x in entry['bbox']]
    cls = entry['class']
    color = colors[cls % len(colors)]
    if entry['image']=='image_1_only':
        cv2.rectangle(img1_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    else:
        cv2.rectangle(img2_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

# Combine side-by-side
diff_img = np.hstack([img1_vis, img2_vis])
cv2.imwrite(diff_img_path, cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))
print(f"Saved object-level diff PNG: {diff_img_path}")