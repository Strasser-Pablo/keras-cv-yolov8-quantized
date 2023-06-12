import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np

calocell_file = "user.cantel.33070624._000001.calocellD3PD_mc16_JZW4.r14423.h5"
jet_file = "user.cantel.33070624._000001.jetD3PD_mc16_JZW4.r14423.h5"

images = np.array(tf.ones(shape=(2, 128, 128, 3)))
images = np.ones((2, 64, 64, 3))
labels = {
    "boxes": np.array([
        [
            [15, 14, 10, 10],
            [20, 39, 10, 10],
            [17, 29, 10, 10],
        ],

        [
            [25, 43, 10, 10],
            [23, 32, 10, 10],
            [32, 32, 10, 10],
        ]
    ]),
    "classes": np.array([[1, 1, 1], [1, 1, 0]]),
}

images = np.load("matrices_training.npy")

labels = {
    "boxes": np.load("labels_training.npy"),
    "classes": np.load("classes_training.npy")
}

# test = np.load("labels_training.npy")
# print(test)

images_validation = np.load("matrices_validation.npy")

labels_validation = {
    "boxes": np.load("labels_validation.npy"),
    "classes": np.load("classes_validation.npy")
}

# print(labels)
# print(len(labels["boxes"][0]))
# print(len(labels["classes"][0]))

model = keras_cv.models.YOLOV8Detector(
    num_classes=2,
    bounding_box_format="center_xywh",
    backbone=keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone"
    ),
    fpn_depth=2
)

# Evaluate model
# model(images)

# Get predictions using the model
# model.predict(images)

# Train model
model.compile(
    classification_loss='binary_crossentropy',
    box_loss='iou',
    optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
    jit_compile=False,
)
model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=6)

print(model.predict(images_validation[:1]))