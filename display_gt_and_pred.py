import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NAME_BACKBONE = "yolo_v8_xs_backbone"
CONFIDENCE = 0.3

images_test = np.load("matrices_test.npy")

print("Nb images : " + str(len(images_test)))

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
results = model.predict(images_test)

for i in range(len(results["boxes"].to_tensor())):
    plt.imshow(images_test[i])

    for j in range(len(results["boxes"][i])):
        if results["classes"][i][j] == 1 and results["confidence"][i][j] >= CONFIDENCE:
            x = results["boxes"][i][j].numpy()[0]-int(round((results["boxes"][i][j].numpy()[2])/2))
            y = results["boxes"][i][j].numpy()[1]-int(round((results["boxes"][i][j].numpy()[3])/2))
            w = results["boxes"][i][j].numpy()[2]
            h = results["boxes"][i][j].numpy()[3]
            plt.gca().add_patch(Rectangle((x, y), w, h, edgecolor="#7736e3", facecolor="none", lw=3))

    for k in range(len(labels_test["boxes"][i])):
        if labels_test["boxes"][i][k][0] < 50:
            x = labels_test["boxes"][i][k][0]-int(round(labels_test["boxes"][i][k][2]/2))
            y = labels_test["boxes"][i][k][1]-int(round(labels_test["boxes"][i][k][3]/2))
            w = labels_test["boxes"][i][k][2]
            h = labels_test["boxes"][i][k][3]
            plt.gca().add_patch(Rectangle((x, y), w, h, edgecolor="red", facecolor="none", lw=3))

    plt.show()

# print("Boxes : ", labels_test["boxes"][:1])
# print("Classes : ", labels_test["classes"][:1])