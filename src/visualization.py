import cv2
import pandas as pd
import pydicom
import numpy as np

# ---------------------------------------
# 1. Paths
# ---------------------------------------

# csv_path = "../dataset/train.csv"
csv_path = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\train.csv"


# dicom_path = (
#     "../dataset/train/"
#     "0005e8e3701dfb1dd93d53e2ff537b6e.dicom"
# )
# dicom_path = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\train"
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

# ---------------------------------------
# 4. Normalize image
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
display_image = image;

# ---------------------------------------
# 5. Convert grayscale to BGR
# ---------------------------------------

# display_image = cv2.cvtColor(
#     image,
#     cv2.COLOR_GRAY2BGR
# )

# ---------------------------------------
# 6. Image ID
# ---------------------------------------

image_id = "0005e8e3701dfb1dd93d53e2ff537b6e"

# ---------------------------------------
# 7. Get annotations
# ---------------------------------------

annotations = df[
    (df["image_id"] == image_id) &
    (df["class_name"] != "No finding")
].copy()

# ---------------------------------------
# 8. Group identical bounding boxes
# ---------------------------------------

grouped = annotations.groupby(
    ["x_min", "y_min", "x_max", "y_max"]
)

# ---------------------------------------
# 9. Draw grouped boxes
# ---------------------------------------

for box, group in grouped:

    x1, y1, x2, y2 = map(int, box)

    # Get all disease names for this box
    labels = group["class_name"].tolist()

    label_text = " + ".join(labels)

    print("\nBox:")
    print(x1, y1, x2, y2)

    print("Labels:")
    print(labels)

    # Draw only ONE rectangle
    cv2.rectangle(
        display_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Draw combined label
    cv2.putText(
        display_image,
        label_text,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

# ---------------------------------------
# 10. Display
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