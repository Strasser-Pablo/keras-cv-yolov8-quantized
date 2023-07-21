import numpy as np
import onnx


def check_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Get the graph from the model
    graph = model.graph

    # Print model metadata
    print("ONNX Model Information:")
    print("Model IR Version:", model.ir_version)
    print("Opset Version:", model.opset_import[0].version)

    # Print input information
    print("\nInput Information:")
    for input_tensor in graph.input:
        print("Name:", input_tensor.name)
        print(
            "Shape:",
            [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim],
        )
        print(
            "Data Type:",
            onnx.TensorProto.DataType.Name(
                input_tensor.type.tensor_type.elem_type
            ),
        )

    # Print output information
    print("\nOutput Information:")
    for output_tensor in graph.output:
        print("Name:", output_tensor.name)
        print(
            "Shape:",
            [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim],
        )
        print(
            "Data Type:",
            onnx.TensorProto.DataType.Name(
                output_tensor.type.tensor_type.elem_type
            ),
        )

    # Print node information
    print("\nNode Information:")
    for node in graph.node:
        print("Name:", node.name)
        print("Op Type:", node.op_type)
        print("Inputs:", node.input)
        print("Outputs:", node.output)

        # Check if the node has constant inputs
        constant_inputs = []
        for input_name in node.input:
            for initializer in graph.initializer:
                if initializer.name == input_name:
                    constant_inputs.append(
                        {
                            "Name": initializer.name,
                            "Data Type": onnx.TensorProto.DataType.Name(
                                initializer.data_type
                            ),
                            "Shape": list(initializer.dims),
                            "Data": np.array(
                                onnx.numpy_helper.to_array(initializer)
                            ),
                        }
                    )

        if constant_inputs:
            print("Constant Inputs:")
            for constant_input in constant_inputs:
                print("Name:", constant_input["Name"])
                print("Data Type:", constant_input["Data Type"])
                print("Shape:", constant_input["Shape"])
                print("Data:")
                print(constant_input["Data"])
                print()


# Provide the path to your ONNX model
model_path = "modified_model.onnx"

# Call the function to check the ONNX model
check_onnx_model(model_path)
