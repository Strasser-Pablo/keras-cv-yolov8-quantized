import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
import os
import sys

NAME_BACKBONE = "yolo_v8_xs_backbone"
CONFIDENCE = 0.5


def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection
    
    iou = intersection / union

    return iou, intersection, union


images_test = np.load("matrices_test.npy")

labels_test = {
    "boxes": np.load("labels_test.npy"),
    "classes": np.load("classes_test.npy")
}

model = keras_cv.models.YOLOV8Detector(
    num_classes=2,
    bounding_box_format="center_xywh",
    backbone=keras_cv.models.YOLOV8Backbone.from_preset(
        NAME_BACKBONE
    ),
    fpn_depth=2
)

if not os.path.isfile(NAME_BACKBONE+".h5"):
    sys.exit(1)

model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

# Get predictions using the model
results = model.predict(images_test[:2])
print(results)
for i in range(len(results["boxes"])):
    boxes = []
    for j in range(len(results["boxes"][i])):
        if results["classes"][i][j] == 1 and results["confidence"][i][j] >= CONFIDENCE:
            boxes.append(results["boxes"][i][j])


# print("Boxes : ", labels_test["boxes"][:1])
# print("Classes : ", labels_test["classes"][:1])