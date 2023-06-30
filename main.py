import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
import os

NAME_BACKBONE = "yolo_v8_xs_backbone"
TRAIN = True

images = np.load("matrices_training.npy")

labels = {
    "boxes": np.load("labels_training.npy"),
    "classes": np.load("classes_training.npy")
}

images_validation = np.load("matrices_validation.npy")

labels_validation = {
    "boxes": np.load("labels_validation.npy"),
    "classes": np.load("classes_validation.npy")
}

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

# Evaluate model
# model(images)

if os.path.isfile(NAME_BACKBONE+".h5") and not TRAIN:
    model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

else:
    # Train model
    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='iou',
        optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=False,
    )
    model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=50, batch_size=32)

    model.save_weights(NAME_BACKBONE+".h5", overwrite="True", save_format="h5", options=None)


# model.summary()
# Get predictions using the model
# print(model.predict(images_test[:1]))
# print("Boxes : ", labels_test["boxes"][:1])
# print("Classes : ", labels_test["classes"][:1])

# tf.keras.saving.save_model(model, "model.tf")
