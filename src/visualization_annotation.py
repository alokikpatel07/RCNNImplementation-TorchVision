import os
import cv2
import pandas as pd
import pydicom
import numpy as np

# ---------------------------------------
# 1. Paths
# ---------------------------------------

# csv_path = "../dataset/train.csv"
csv_path = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\train.csv"
# dicom_path = "../dataset/train/0005e8e3701dfb1dd93d53e2ff537b6e.dicom"
dicom_path = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\train\0005e8e3701dfb1dd93d53e2ff537b6e.dicom"

# ---------------------------------------
# 2. Read CSV
# ---------------------------------------

df = pd.read_csv(csv_path)

# ---------------------------------------
# 3. Read DICOM
# ---------------------------------------

ds = pydicom.dcmread(dicom_path)

image = ds.pixel_array

print("Original image shape:", image.shape)
print("Original data type:", image.dtype)

# ---------------------------------------
# 4. Convert image to displayable format
# ---------------------------------------

image = image.astype(np.float32)

image = cv2.normalize(
    image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

image = image.astype(np.uint8)

# ---------------------------------------
# 5. Convert grayscale → BGR
# ---------------------------------------

display_image = cv2.cvtColor(
    image,
    cv2.COLOR_GRAY2BGR
)

# ---------------------------------------
# 6. Get image ID
# ---------------------------------------

filename = os.path.basename(dicom_path)

image_id = os.path.splitext(filename)[0]

print("Image ID:", image_id)

# ---------------------------------------
# 7. Get annotations
# ---------------------------------------

annotations = df[df["image_id"] == image_id]

print("\nAnnotations:")
print(annotations)

# ---------------------------------------
# 8. Draw bounding boxes
# ---------------------------------------

for _, row in annotations.iterrows():

    # Ignore "No finding"
    if row["class_name"] == "No finding":
        continue

    # Get coordinates
    x1 = int(row["x_min"])
    y1 = int(row["y_min"])
    x2 = int(row["x_max"])
    y2 = int(row["y_max"])

    class_name = row["class_name"]

    # Draw rectangle
    cv2.rectangle(
        display_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Draw class name
    cv2.putText(
        display_image,
        class_name,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# ---------------------------------------
# 9. Display image
# ---------------------------------------

cv2.namedWindow(
    "Medical Image",
    cv2.WINDOW_NORMAL
)

cv2.imshow(
    "Medical Image",
    display_image
)

cv2.waitKey(0)

cv2.destroyAllWindows()