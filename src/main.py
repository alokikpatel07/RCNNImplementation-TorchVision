# ============================================================
# LOCAL FASTER R-CNN MEDICAL IMAGE DETECTION
# Uses the trained .pth model
#
# Supported input:
#   - DICOM (.dicom / .dcm)
#   - PNG
#   - JPG / JPEG
#
# The program:
#   1. Selects an image from your computer
#   2. Loads the trained .pth model
#   3. Detects abnormalities
#   4. Displays detections with bounding boxes
#   5. Saves the annotated image
# ============================================================

import os
import sys
import cv2
import torch
import pydicom
import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\AP - Projects\RCNN Implementation - TorchVision\models\fasterrcnn_vinbigdata.pth";


CONFIDENCE_THRESHOLD = 0.50

OUTPUT_FOLDER = r"D:\AP - Projects\RCNN Implementation - TorchVision\results"


# ============================================================
# 2. CLASS DEFINITIONS
# ============================================================

# Faster R-CNN uses 0 for background.
# The trained model has 14 medical abnormality classes.

DEFAULT_ID_TO_CLASS = {
    0: "Background",
    1: "Aortic enlargement",
    2: "Atelectasis",
    3: "Calcification",
    4: "Cardiomegaly",
    5: "Consolidation",
    6: "ILD",
    7: "Infiltration",
    8: "Lung Opacity",
    9: "Nodule/Mass",
    10: "Other lesion",
    11: "Pleural effusion",
    12: "Pleural thickening",
    13: "Pneumothorax",
    14: "Pulmonary fibrosis"
}


# ============================================================
# 3. SELECT COMPUTING DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


# ============================================================
# 4. CHECK MODEL FILE
# ============================================================

if not os.path.isfile(MODEL_PATH):

    print("\nERROR:")
    print("Trained model file was not found.")
    print("\nExpected file:")
    print(os.path.abspath(MODEL_PATH))

    sys.exit(1)


# ============================================================
# 5. LOAD CHECKPOINT
# ============================================================

print("\nLoading trained model...")

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

except Exception as e:

    print("\nERROR: Could not load the .pth file.")
    print(e)

    sys.exit(1)


# ============================================================
# 6. GET MODEL INFORMATION
# ============================================================

num_classes = checkpoint.get(
    "num_classes",
    15
)

id_to_class = checkpoint.get(
    "id_to_class",
    DEFAULT_ID_TO_CLASS
)

# Make sure dictionary keys are integers
id_to_class = {
    int(key): value
    for key, value in id_to_class.items()
}


# ============================================================
# 7. RECREATE EXACT FASTER R-CNN ARCHITECTURE
# ============================================================

print("Creating Faster R-CNN architecture...")

model = fasterrcnn_resnet50_fpn(
    weights=None
)

in_features = (
    model.roi_heads.box_predictor
    .cls_score.in_features
)

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)


# ============================================================
# 8. LOAD TRAINED WEIGHTS
# ============================================================

try:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

except Exception as e:

    print("\nERROR:")
    print("The .pth file does not match the Faster R-CNN architecture.")
    print(e)

    sys.exit(1)


model = model.to(device)

model.eval()

print("Trained model loaded successfully.")


# ============================================================
# 9. SELECT IMAGE
# ============================================================

# print("\nSelect a DICOM or image file.")

# root = Tk()
# root.withdraw()

# file_path = filedialog.askopenfilename(
#     title="Select Medical Image",
#     filetypes=[
#         (
#             "Medical/Image files",
#             "*.dicom *.dcm *.png *.jpg *.jpeg"
#         ),
#         (
#             "DICOM files",
#             "*.dicom *.dcm"
#         ),
#         (
#             "Image files",
#             "*.png *.jpg *.jpeg"
#         ),
#         (
#             "All files",
#             "*.*"
#         )
#     ]
# )

# root.destroy()
file_path = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\test\2.dicom";

# ============================================================
# 10. CHECK FILE SELECTION
# ============================================================

if not file_path:

    print("\nNo file selected.")
    sys.exit(0)


print("\nSelected file:")
print(file_path)


# ============================================================
# 11. LOAD IMAGE
# ============================================================

extension = os.path.splitext(
    file_path
)[1].lower()


try:

    if extension in [".dcm", ".dicom"]:

        # ----------------------------------------------------
        # DICOM
        # ----------------------------------------------------

        ds = pydicom.dcmread(
            file_path
        )

        image = ds.pixel_array.astype(
            np.float32
        )

    else:

        # ----------------------------------------------------
        # PNG / JPG / JPEG
        # ----------------------------------------------------

        image = cv2.imread(
            file_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise ValueError(
                "OpenCV could not read the selected image."
            )

        image = image.astype(
            np.float32
        )


except Exception as e:

    print("\nERROR: Could not read the image.")
    print(e)

    sys.exit(1)


# ============================================================
# 12. CHECK IMAGE
# ============================================================

if image is None or image.size == 0:

    print("\nERROR: Image is empty.")

    sys.exit(1)


print("\nOriginal image size:")
print(image.shape)


# ============================================================
# 13. HANDLE IMAGE ORIENTATION / PIXEL RANGE
# ============================================================

# Normalize the image to 0-255 for model input.

image = cv2.normalize(
    image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

image = image.astype(
    np.uint8
)


# ============================================================
# 14. CONVERT GRAYSCALE TO 3 CHANNELS
# ============================================================

image_rgb = np.stack(
    [image, image, image],
    axis=-1
)


# ============================================================
# 15. CONVERT TO PYTORCH TENSOR
# ============================================================

image_tensor = torch.from_numpy(
    image_rgb
).permute(
    2,
    0,
    1
).float() / 255.0


# ============================================================
# 16. RUN MODEL
# ============================================================

print("\nRunning Faster R-CNN detection...")

with torch.no_grad():

    prediction = model(
        [image_tensor.to(device)]
    )[0]


# ============================================================
# 17. GET RAW PREDICTIONS
# ============================================================

boxes = prediction["boxes"].detach().cpu()
labels = prediction["labels"].detach().cpu()
scores = prediction["scores"].detach().cpu()


print("\nRaw predictions:")
print("Boxes :", len(boxes))
print("Labels:", len(labels))
print("Scores:", len(scores))


# ============================================================
# 18. APPLY CONFIDENCE THRESHOLD
# ============================================================

keep = scores >= CONFIDENCE_THRESHOLD

boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]


# ============================================================
# 19. PRINT DETECTION RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("DETECTION RESULTS")
print("=" * 60)

print(
    "Confidence threshold:",
    CONFIDENCE_THRESHOLD
)

print(
    "Detected abnormalities:",
    len(boxes)
)

print("=" * 60)


if len(boxes) == 0:

    print(
        "\nNo abnormality detected above "
        f"{CONFIDENCE_THRESHOLD:.0%} confidence."
    )

else:

    for i in range(len(boxes)):

        class_id = int(
            labels[i].item()
        )

        class_name = id_to_class.get(
            class_id,
            f"Unknown class ({class_id})"
        )

        confidence = float(
            scores[i].item()
        )

        box = boxes[i].tolist()

        print(
            f"\n{i + 1}. {class_name}"
        )

        print(
            f"   Confidence : {confidence:.2%}"
        )

        print(
            f"   Bounding box: "
            f"[{box[0]:.1f}, "
            f"{box[1]:.1f}, "
            f"{box[2]:.1f}, "
            f"{box[3]:.1f}]"
        )


# ============================================================
# 20. CREATE DISPLAY IMAGE
# ============================================================

display_image = image_rgb.copy()


# ============================================================
# 21. DRAW DETECTION BOXES
# ============================================================

for box, label, score in zip(
    boxes,
    labels,
    scores
):

    x_min, y_min, x_max, y_max = [
        int(round(value))
        for value in box.tolist()
    ]

    class_id = int(
        label.item()
    )

    class_name = id_to_class.get(
        class_id,
        f"Unknown ({class_id})"
    )

    confidence = float(
        score.item()
    )

    # --------------------------------------------------------
    # Keep coordinates inside image
    # --------------------------------------------------------

    height, width = image.shape[:2]

    x_min = max(
        0,
        min(x_min, width - 1)
    )

    x_max = max(
        0,
        min(x_max, width - 1)
    )

    y_min = max(
        0,
        min(y_min, height - 1)
    )

    y_max = max(
        0,
        min(y_max, height - 1)
    )

    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        display_image,
        (x_min, y_min),
        (x_max, y_max),
        (255, 0, 0),
        4
    )

    # --------------------------------------------------------
    # Detection label
    # --------------------------------------------------------

    text = (
        f"{class_name}: "
        f"{confidence:.2f}"
    )

    text_y = max(
        30,
        y_min - 10
    )

    cv2.putText(
        display_image,
        text,
        (x_min, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# 22. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 23. SAVE ANNOTATED IMAGE
# ============================================================

input_name = os.path.splitext(
    os.path.basename(file_path)
)[0]

output_path = os.path.join(
    OUTPUT_FOLDER,
    input_name + "_detected.png"
)


# OpenCV expects BGR when saving.
display_bgr = cv2.cvtColor(
    display_image,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    output_path,
    display_bgr
)


# ============================================================
# 24. DISPLAY RESULT
# ============================================================

plt.figure(
    figsize=(12, 12)
)

plt.imshow(
    display_image
)

plt.title(
    "Faster R-CNN Abnormality Detection"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 25. FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("DETECTION COMPLETED")
print("=" * 60)

print(
    "Input :",
    file_path
)

print(
    "Output:",
    os.path.abspath(output_path)
)

print(
    "Detections:",
    len(boxes)
)

print("=" * 60)