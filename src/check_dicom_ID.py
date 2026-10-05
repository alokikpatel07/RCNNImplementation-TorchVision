import os
import pandas as pd

# -----------------------------
# 1. Paths
# -----------------------------
# csv_path = "../dataset/train.csv"
csv_path = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\train.csv"
# dicom_folder = "../dataset/train"
dicom_folder = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\train"

# -----------------------------
# 2. Read CSV
# -----------------------------
df = pd.read_csv(csv_path)

# -----------------------------
# 3. Get DICOM files
# -----------------------------
dicom_files = [
    file for file in os.listdir(dicom_folder)
    if file.lower().endswith(".dicom")
]

print("DICOM files found:", len(dicom_files))

# -----------------------------
# 4. Check each DICOM
# -----------------------------
for file in dicom_files:

    # Remove .dicom
    image_id = os.path.splitext(file)[0]

    # Find annotations
    annotations = df[df["image_id"] == image_id]

    print("\n--------------------------------")
    print("DICOM:", file)
    print("Image ID:", image_id)
    print("Annotations:", len(annotations))

    if len(annotations) == 0:
        print("WARNING: Image ID not found in train.csv")
    else:
        print(annotations[
            ["class_name", "class_id",
            "x_min", "y_min", "x_max", "y_max"]
        ])