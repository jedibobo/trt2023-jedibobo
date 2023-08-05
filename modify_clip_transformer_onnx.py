import onnx_graphsurgeon as gs
import onnx
import numpy as np

# load sd_clip_transformer.onnx 
graph = gs.import_onnx(onnx.load("sd_clip_transformer.onnx"))

# modify the shape of output to (1, 77, 768)
output_tensor = graph.outputs[0]    
output_tensor.shape = ["B", 77, 768]

output_tensor = graph.outputs[1]
output_tensor.shape = ["B", 768]

# find a node named /text/ArgMax
# node = graph.find_node_by_name("/text/ArgMax")
# # add a cast node before this node, converting from int64 to int32
# cast_node = gs.Node()

# save the modified model to sd_clip_transformer_onnx_reshape.onnx
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "sd_clip_transformer_onnx_reshape.onnx")


from polygraphy.backend.onnx import modify_outputs, onnx_from_path
new_onnx_path = "sd_clip_transformer_onnx_reshape.onnx"
new_onnx_model = onnx_from_path(new_onnx_path)
# get onnx runtime output tensor info(name, type, shape)
# change onnx -inf to -1e4
for node in new_onnx_model.graph.node:
    # if node.name == "/text_model/ConstantOfShape_1":
    if node.op_type == "ConstantOfShape":
        print(node)
        attr = node.attribute[0]
        print(attr)
        if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
            np_array = np.frombuffer(attr.t.raw_data, dtype=np.float32).copy()
            print("raw array", np_array)
            np_array[np_array == -np.inf] = -100000  # 将所有负无穷的值改为-100000
            attr.t.raw_data = np_array.tobytes() 
            print("new array", np_array)
        print(attr)
onnx.save_model(
    new_onnx_model,
    "./sd_clip_transformer_reshape.onnx",
)

