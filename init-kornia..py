import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import pytesseract

root = r"E:\boss-gpt"
path1 = os.path.join(root, "images", "image-1.webp")
path2 = os.path.join(root, "images", "image-2.webp")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_REGION_AREA = 500
OUTPUT_FOLDER = os.path.join(root, "diff_regions")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def load_images(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2

def get_diff_regions(img1, img2):
    """Return regions with visual differences (missing objects or color changes)"""

    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    gray1 = lab1[:,:,0]
    gray2 = lab2[:,:,0]
    score, diff_struct = ssim(gray1, gray2, full=True)
    diff_struct = (diff_struct * 255).astype(np.uint8)

    diff_color = cv2.absdiff(lab1[:,:,1:], lab2[:,:,1:])
    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)

    combined = cv2.bitwise_or(
        cv2.threshold(diff_struct, 200, 255, cv2.THRESH_BINARY_INV)[1],
        cv2.threshold(diff_color, 25, 255, cv2.THRESH_BINARY)[1]
    )


    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h > MIN_REGION_AREA:
            regions.append((x, y, w, h))
    return regions, score

def extract_text_from_regions(img, regions):
    """Run OCR on regions. Returns DataFrame with text."""
    results = []
    for i, (x, y, w, h) in enumerate(regions):
        crop = img[y:y+h, x:x+w]
        text = pytesseract.image_to_string(crop).strip()
        results.append({
            "region_id": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "text": text
        })
    return pd.DataFrame(results)

def classify_region(row):
    """Classify region type based on OCR text"""
    if pd.isna(row["text_img1"]) and pd.isna(row["text_img2"]):
        return "object difference"
    elif row["text_img1"] != row["text_img2"]:
        return "text difference"
    else:
        return "object difference"

def save_diff_regions(img, regions):
    """Save cropped difference regions"""
    for i, (x, y, w, h) in enumerate(regions):
        crop = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"region_{i}.png"), crop)

# -----------------------------
# MAIN
# -----------------------------
def main(path1, path2):
    img1, img2 = load_images(path1, path2)
    regions, score = get_diff_regions(img1, img2)
    print(f"Similarity score: {score:.4f}, Detected {len(regions)} changed regions")

    df1 = extract_text_from_regions(img1, regions)
    df2 = extract_text_from_regions(img2, regions)

    result = pd.merge(df1, df2, on="region_id", how="outer", suffixes=("_img1", "_img2"))
    result["changed"] = result["text_img1"] != result["text_img2"]

    result["region_type"] = result.apply(classify_region, axis=1)

    result.to_csv(os.path.join(root, "diff_report.csv"), index=False)
    result.to_excel(os.path.join(root, "diff_report.xlsx"), index=False)
    print("Saved reports to CSV and Excel")

    # Visual diff
    vis = img2.copy()
    for (x, y, w, h) in regions:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,0,255), 2)
    diff_path = os.path.join(root, "diff_visual.png")
    cv2.imwrite(diff_path, vis)
    print("Saved visual diff to:", diff_path)

    # Save individual regions
    save_diff_regions(img2, regions)
    print(f"Saved {len(regions)} cropped regions to {OUTPUT_FOLDER}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main(path1, path2)