import onnx
from onnx import numpy_helper
from onnx.helper import make_node
from onnx.helper import make_tensor_value_info


def replace_concat_with_multiple_concats_and_fix_shape(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Get the graph from the model
    graph = model.graph

    # Find the Concat nodes with more than two inputs
    concat_nodes = [
        node
        for node in graph.node
        if node.op_type == "Concat" and len(node.input) > 2
    ]

    for concat_node in concat_nodes:
        inputs = concat_node.input

        # Determine the number of new Concat nodes needed
        num_concats = len(inputs) // 2

        # Create new Concat nodes with two inputs each
        new_concat_inputs = []
        for i in range(num_concats):
            concat_inputs = inputs[i * 2 : (i + 1) * 2]
            new_concat_name = f"new_concat_{concat_node.name}_{i}"
            new_concat_output = f"new_concat_output_{concat_node.name}_{i}"
            new_concat_node = make_node(
                "Concat",
                concat_inputs,
                [new_concat_output],
                name=new_concat_name,
            )
            new_concat_node.attribute.extend(concat_node.attribute)
            graph.node.extend([new_concat_node])
            new_concat_inputs.append(new_concat_output)

        # Adjust connections between nodes
        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == concat_node.output[0]:
                    # Replace original Concat node output with new Concat node outputs
                    node.input[i] = new_concat_inputs[0]
                    for j in range(1, num_concats):
                        new_output = f"new_concat_output_{concat_node.name}_{j}"
                        node.input.insert(i + j, new_output)
        graph.node.remove(concat_node)

        # Update the shape information of the concatenated input tensors
        for i in range(num_concats):
            new_concat_input_tensors = [
                tensor
                for tensor in graph.input
                if tensor.name == new_concat_inputs[i]
            ]

            total_dim = sum(
                [
                    tensor.type.tensor_type.shape.dim[
                        concat_node.attribute[0].ints[0]
                    ]
                    for tensor in new_concat_input_tensors
                ]
            )

            for tensor in new_concat_input_tensors:
                tensor.type.tensor_type.shape.dim[
                    concat_node.attribute[0].ints[0]
                ].dim_value = total_dim

    # Save the modified model
    onnx.save(model, "modified_model.onnx")


# Provide the path to your ONNX model
model_path = "model.onnx"

# Call the function to replace Concat node with multiple Concat nodes and fix tensor shape
replace_concat_with_multiple_concats_and_fix_shape(model_path)
