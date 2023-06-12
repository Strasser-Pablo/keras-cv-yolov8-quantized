import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np

# images = tf.ones(shape=(1, 512, 512, 3))
# labels = {
#     "boxes": [
#         [
#             [0, 0, 100, 100],
#             [100, 100, 200, 200],
#             [300, 300, 100, 100],
#         ]
#     ],
#     "classes": [[1, 1, 1]],
# }

# backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone")
# # print(backbone)
# model = keras_cv.models.YOLOV8Detector(backbone=backbone, num_classes=1, bounding_box_format="xywh")
# # print(model.summary())

# # print(model(images))

# # print(model.predict(images))

# model.compile(
#     classification_loss='binary_crossentropy',
#     box_loss='iou',
#     optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
#     jit_compile=False,
# )
# model.fit(images, labels)

# print(model.predict(images))


images = tf.ones(shape=(1, 512, 512, 3))
labels = {
    "boxes": np.array([
        [
            [0, 0, 100, 100],
            [100, 100, 200, 200],
            [300, 300, 100, 100],
        ]
    ]),
    "classes": np.array([[1, 1, 1]]),
}
model = keras_cv.models.YOLOV8Detector(
    num_classes=20,
    bounding_box_format="xywh",
    backbone=keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone"
    ),
    fpn_depth=2
)

# Evaluate model
model(images)

# Get predictions using the model
model.predict(images)

# Train model
model.compile(
    classification_loss='binary_crossentropy',
    box_loss='iou',
    optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
    jit_compile=False,
)
model.fit(images, labels)