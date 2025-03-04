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

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_cv.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNet50V2Backbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNetV2Backbone,
)
from keras_cv.utils.train import get_feature_extractor


class ResNetV2BackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = tf.ones(shape=(8, 224, 224, 3))

    def test_valid_call(self):
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = ResNet50V2Backbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=True,
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, ResNetV2Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_alias_model(self, save_format, filename):
        model = ResNet50V2Backbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, ResNetV2Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_create_backbone_model_from_alias_model(self):
        model = ResNet50V2Backbone(
            include_rescaling=False,
        )
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        inputs = keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        # Resnet50 backbone has 4 level of features (P2 ~ P5)
        levels = ["P2", "P3", "P4", "P5"]
        self.assertLen(outputs, 4)
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(outputs["P2"].shape, [None, 64, 64, 256])
        self.assertEquals(outputs["P3"].shape, [None, 32, 32, 512])
        self.assertEquals(outputs["P4"].shape, [None, 16, 16, 1024])
        self.assertEquals(outputs["P5"].shape, [None, 8, 8, 2048])

    def test_create_backbone_model_with_level_config(self):
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
            input_shape=[256, 256, 3],
        )
        levels = ["P3", "P4"]
        layer_names = [model.pyramid_level_inputs[level] for level in levels]
        backbone_model = get_feature_extractor(model, layer_names, levels)
        inputs = keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 2)
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(outputs["P3"].shape, [None, 32, 32, 512])
        self.assertEquals(outputs["P4"].shape, [None, 16, 16, 1024])

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        # ResNet50 model
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[3, 4, 6, 3],
            stackwise_strides=[1, 2, 2, 2],
            input_shape=(None, None, num_channels),
            include_rescaling=False,
        )
        self.assertEqual(model.output_shape, (None, None, None, 2048))


if __name__ == "__main__":
    tf.test.main()
