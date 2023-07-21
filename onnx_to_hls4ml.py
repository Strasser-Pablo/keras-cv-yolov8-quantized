import hls4ml
import keras
import numpy as np
import onnx
import tensorflow as tf
from tensorflow.keras.models import load_model

import keras_cv

model = onnx.load("modified_model.onnx")

config = hls4ml.utils.config_from_onnx_model(model)

print(config)


print("-----------------------------------")
print("Configuration")
print(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_onnx_model(
    model,
    hls_config=config,
    output_dir="model_1/hls4ml_prj",
    part="xcu250-figd2104-2L-e",
)

hls4ml.utils.plot_model(
    hls_model, show_shapes=True, show_precision=True, to_file=None
)

hls_model.compile()

# # You can print the configuration to see some default parameters
# print(config)

# # Convert it to a hls project
# hls_model = hls4ml.converters.keras_to_hls(config)

# # Print full list of example models if you want to explore more
# hls4ml.utils.fetch_example_list()

# # Use Vivado HLS to synthesize the model
# # This might take several minutes
# hls_model.build()

# # Print out the report if you want
# hls4ml.report.read_vivado_report('my-hls-test')
