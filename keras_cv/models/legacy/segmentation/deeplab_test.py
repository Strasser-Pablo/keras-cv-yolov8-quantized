# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from keras_cv.models import ResNet50V2Backbone
from keras_cv.models.legacy.segmentation.deeplab import DeepLabV3


class DeeplabTest(tf.test.TestCase):
    def test_deeplab_model_construction_with_preconfigured_setting(self):
        backbone = ResNet50V2Backbone(
            input_shape=[64, 64, 3],
        )
        model = DeepLabV3(num_classes=11, backbone=backbone)
        input_image = tf.random.uniform(shape=[1, 64, 64, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].shape, [1, 64, 64, 11])

    def test_greyscale_input(self):
        backbone = ResNet50V2Backbone(
            input_shape=[64, 64, 1],
        )
        model = DeepLabV3(num_classes=11, backbone=backbone)
        input_image = tf.random.uniform(shape=[1, 64, 64, 1])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].shape, [1, 64, 64, 11])

    def test_missing_input_shapes(self):
        with self.assertRaisesRegex(
            ValueError,
            "Input shapes for both the backbone and DeepLabV3 "
            "cannot be `None`.",
        ):
            backbone = ResNet50V2Backbone()
            DeepLabV3(num_classes=11, backbone=backbone)

    def test_deeplab_model_with_components(self):
        backbone = ResNet50V2Backbone(
            input_shape=[64, 64, 3],
        )
        model = DeepLabV3(
            num_classes=11,
            backbone=backbone,
        )

        input_image = tf.random.uniform(shape=[1, 64, 64, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].shape, [1, 64, 64, 11])

    def test_mixed_precision(self):
        keras.mixed_precision.set_global_policy("mixed_float16")
        backbone = ResNet50V2Backbone(
            input_shape=[64, 64, 3],
        )
        model = DeepLabV3(
            num_classes=11,
            backbone=backbone,
        )
        input_image = tf.random.uniform(shape=[1, 64, 64, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].dtype, tf.float32)
        keras.mixed_precision.set_global_policy("float32")

    def test_invalid_backbone_model(self):
        with self.assertRaisesRegex(
            ValueError, "Argument `backbone` must be a `keras.layers.Layer`"
        ):
            DeepLabV3(
                num_classes=11,
                backbone=tf.Module(),
            )

    @pytest.mark.extra_large
    def test_model_train(self):
        target_size = [96, 96]

        backbone = ResNet50V2Backbone(
            input_shape=target_size + [3],
        )
        model = DeepLabV3(num_classes=1, backbone=backbone)

        gcs_data_pattern = "gs://caltech_birds2011_mask/0.1.1/*.tfrecord*"
        features = tfds.features.FeaturesDict(
            {
                "bbox": tfds.features.BBoxFeature(),
                "image": tfds.features.Image(
                    shape=(None, None, 3), dtype=tf.uint8
                ),
                "image/filename": tfds.features.Text(),
                "label": tfds.features.ClassLabel(num_classes=200),
                "label_name": tfds.features.Text(),
                "segmentation_mask": tfds.features.Image(
                    shape=(None, None, 1), dtype=tf.uint8
                ),
            }
        )

        filenames = tf.io.gfile.glob(gcs_data_pattern)
        AUTO = tf.data.AUTOTUNE
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        ds = ds.with_options(ignore_order)
        ds = ds.map(features.deserialize_example, num_parallel_calls=AUTO)

        resizing = keras.layers.Resizing(target_size[1], target_size[0])

        def resize_images_and_masks(data):
            image = tf.image.convert_image_dtype(
                data["image"], dtype=tf.float32
            )
            data["image"] = resizing(image)
            # WARNING: assumes processing unbatched
            mask = data["segmentation_mask"]
            mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
            data["segmentation_mask"] = resizing(mask)
            return data

        def keep_image_and_mask_only(data):
            return data["image"], data["segmentation_mask"]

        dataset = ds
        dataset = dataset.map(resize_images_and_masks)
        dataset = dataset.map(keep_image_and_mask_only)

        batch_size = 32
        training_dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size)
        )
        training_dataset = training_dataset.repeat()

        epochs = 1
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        model.fit(
            training_dataset.take(10),
            epochs=epochs,
        )


if __name__ == "__main__":
    tf.test.main()
